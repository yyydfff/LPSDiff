import os
import sys
import copy
import inspect
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse
from models.basic_template import TrainTask
from models import model_dict

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None


def _is_dataloader(obj):
    return isinstance(obj, DataLoader) or (
        hasattr(obj, "__iter__") and hasattr(obj, "__len__")
        and not isinstance(obj, (torch.Tensor, nn.Module))
    )


def _scan_container_for_loader(container, found):
    if _is_dataloader(container):
        found.append(container)
    elif isinstance(container, (list, tuple, set)):
        for v in container:
            _scan_container_for_loader(v, found)
    elif isinstance(container, dict):
        for v in container.values():
            _scan_container_for_loader(v, found)


def _find_loader(model):
    common_names = [
        "train_loader", "train_dataloader", "tr_loader",
        "data_loader", "dataloader", "loader",
        "val_loader", "valid_loader", "validation_loader",
        "test_loader", "eval_loader"
    ]
    for name in common_names:
        if hasattr(model, name):
            obj = getattr(model, name)
            if _is_dataloader(obj):
                return obj
    candidates = []
    for _, v in vars(model).items():
        _scan_container_for_loader(v, candidates)
    if candidates:
        return candidates[0]
    raise NameError("_find_loader 没找到可用的 DataLoader")


def _first_tensor(batch):
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)):
        for item in batch:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(batch, dict):
        for key in ["image", "images", "input", "inputs", "x", "data"]:
            if key in batch and torch.is_tensor(batch[key]):
                return batch[key]
        for v in batch.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _second_tensor(batch):
    def _iter_tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                yield from _iter_tensors(it)
        elif isinstance(obj, dict):
            for key in ["label","labels","y","target","targets"]:
                if key in obj and torch.is_tensor(obj[key]):
                    yield obj[key]
            for v in obj.values():
                yield from _iter_tensors(v)

    it = _iter_tensors(batch)
    try:
        next(it)
        second = next(it)
        return second
    except StopIteration:
        return None


def _find_label_from_batch(batch):
    if isinstance(batch, dict):
        for key in ["label","labels","y","target","targets"]:
            if key in batch and torch.is_tensor(batch[key]):
                return batch[key]
    t2 = _second_tensor(batch)
    return t2


def _to_hw(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return (int(value[-2]), int(value[-1]))
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
    return (224, 224)


def _guess_dummy_input(opt, device):
    B = int(getattr(opt, "batch_size", 1))
    C = int(getattr(opt, "in_channels", getattr(opt, "num_channels", 3)))
    size_keys = ["image_size", "img_size", "input_size", "size", "resolution"]
    H, W = 224, 224
    for k in size_keys:
        if hasattr(opt, k):
            H, W = _to_hw(getattr(opt, k))
            break
    D = getattr(opt, "clip_len", getattr(opt, "depth", None))
    if D is not None:
        D = int(D)
        return torch.zeros(B, C, D, H, W, device=device, dtype=torch.float32)
    return torch.zeros(B, C, H, W, device=device, dtype=torch.float32)


def _is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _unwrap_module(obj):
    if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return obj.module
    m = getattr(obj, "module", None)
    return m if isinstance(m, nn.Module) else obj


def _flatten_modules(obj, found, visited, max_depth=3, depth=0):
    if obj is None:
        return
    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    if isinstance(obj, nn.Module):
        found.add(obj)
        for child in obj.children():
            _flatten_modules(child, found, visited, max_depth, depth + 1)
        return

    if depth >= max_depth:
        return

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _flatten_modules(v, found, visited, max_depth, depth + 1)
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_modules(v, found, visited, max_depth, depth + 1)
        return

    try:
        for _, v in vars(obj).items():
            _flatten_modules(v, found, visited, max_depth, depth + 1)
    except Exception:
        pass


def _gather_unique_params(modules):
    seen = set()
    total = 0
    for m in modules:
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                total += p.numel()
    return total


def _guess_primary_module(root):
    root = _unwrap_module(root)

    if isinstance(root, nn.Module):
        return root

    priority_names = [
        "model","net","backbone","backbone_net",
        "unet","unet_model","generator","G",
        "student","teacher","encoder","decoder",
        "diffusion","core","module","wrapper"
    ]

    for name in priority_names:
        if hasattr(root, name):
            cand = getattr(root, name)
            cand = _unwrap_module(cand)
            if isinstance(cand, nn.Module):
                return cand

    found = set()
    _flatten_modules(root, found, set(), max_depth=2)

    if len(found) == 1:
        return list(found)[0]
    return None


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k:_to_device(v,device) for k,v in obj.items()}
    return obj


def _build_flops_inputs(primary, x, batch=None, opt=None):
    B = x.shape[0] if torch.is_tensor(x) and x.dim()>0 else 1
    y_from_batch = _find_label_from_batch(batch) if batch is not None else None

    if y_from_batch is None:
        try:
            y_from_batch = torch.zeros(B,dtype=torch.long,device=x.device)
        except Exception:
            y_from_batch = 0
    elif torch.is_tensor(y_from_batch):
        y_from_batch = y_from_batch.to(x.device, non_blocking=True)

    n_iter_val = int(getattr(opt,"resume_iter",0)) if opt is not None else 0

    name2val = {
        "x":x,"inp":x,"input":x,"inputs":x,"image":x,"images":x,
        "y":y_from_batch,"label":y_from_batch,"labels":y_from_batch,
        "target":y_from_batch,"targets":y_from_batch,
        "n_iter":n_iter_val,"iter":n_iter_val,"step":n_iter_val,"global_step":n_iter_val,
    }

    try:
        sig = inspect.signature(primary.forward)
    except Exception:
        return x

    params=[p for p in sig.parameters.values() if p.name!="self"]

    if not params:
        return x

    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD):
            return x

    args=[]
    for p in params:
        pname=p.name
        if pname in name2val:
            args.append(name2val[pname])
        else:
            if p.default is inspect._empty:
                raise ValueError(f"缺少必须入参 '{pname}'")

    if len(args)==1:
        return _to_device(args[0],x.device)
    return _to_device(tuple(args),x.device)


def safe_print_flops_params(obj,x,batch=None,opt=None):
    modules=set()
    _flatten_modules(obj,modules,set(),max_depth=3)

    if not modules:
        if _is_main_process():
            print("[Params] 没有找到 nn.Module")
        return

    params=_gather_unique_params(modules)

    if FlopCountAnalysis is None:
        if _is_main_process():
            print(f"Params: {params/1e6:.3f} M")
        return

    primary=_guess_primary_module(obj)
    if primary is None:
        return

    try:
        primary.eval().to(x.device)
    except Exception:
        pass

    try:
        inputs_for_forward=_build_flops_inputs(primary,x,batch,opt)
        inputs_for_forward=_to_device(inputs_for_forward,x.device)

        flops=FlopCountAnalysis(primary,inputs_for_forward).total()

        if _is_main_process():
            print(f"GFLOPs: {flops/1e9:.3f} | Params: {params/1e6:.3f} M")

    except Exception as e:
        msg=str(e).lower()

        if "same device" in msg or "found at least two devices" in msg:
            try:
                primary_cpu=copy.deepcopy(primary).cpu()
                inputs_cpu=_to_device(inputs_for_forward,torch.device("cpu"))
                flops=FlopCountAnalysis(primary_cpu,inputs_cpu).total()

                if _is_main_process():
                    print(f"GFLOPs(CPU): {flops/1e9:.3f} | Params:{params/1e6:.3f}M")
                return
            except Exception:
                return


if __name__=="__main__":
    default_parser=TrainTask.build_default_options()
    default_opt,unknown_opt=default_parser.parse_known_args()

    if default_opt.model_name in model_dict:
        MODEL_CLASS=model_dict[default_opt.model_name]
    else:
        sys.exit(1)

    if hasattr(MODEL_CLASS,'build_options'):
        private_parser=MODEL_CLASS.build_options()
    else:
        private_parser=argparse.ArgumentParser()

    opt=private_parser.parse_args(unknown_opt,namespace=default_opt)

    if hasattr(opt,'gpu_id') and opt.gpu_id is not None:
        torch.cuda.set_device(opt.gpu_id)
        device=torch.device(f'cuda:{opt.gpu_id}')
    else:
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    model=MODEL_CLASS(opt)

    model.fit()