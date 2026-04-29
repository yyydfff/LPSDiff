import os
import argparse
import torch
import torch.nn as nn

from models.basic_template import TrainTask
from models import model_dict


def _parse_all_args():
    default_parser=TrainTask.build_default_options()
    default_opt,unknown_opt=default_parser.parse_known_args()

    if default_opt.model_name not in model_dict:
        raise ValueError(
            f"未知模型: {default_opt.model_name}. 可选: {list(model_dict.keys())}"
        )

    MODEL_CLASS=model_dict[default_opt.model_name]

    private_parser=(
        MODEL_CLASS.build_options()
        if hasattr(MODEL_CLASS,'build_options')
        else argparse.ArgumentParser()
    )

    private_parser.add_argument(
        '--ckpt',
        type=str,
        default='test_results/best.model'
    )
    private_parser.add_argument(
        '--test_iter',
        type=int,
        default=0
    )
    private_parser.add_argument(
        '--epoch_for_log',
        type=int,
        default=0
    )

    opt=private_parser.parse_args(
        unknown_opt,
        namespace=default_opt
    )

    opt.mode='test'

    if not hasattr(opt,'log_file') or not opt.log_file:
        opt.log_file='test.log'

    if not hasattr(opt,'log_mode') or not opt.log_mode:
        opt.log_mode='append'

    return opt,MODEL_CLASS


def _infer_default_best_path():
    project_root=os.path.abspath(os.path.dirname(__file__))
    return os.path.join(
        project_root,
        'test_results',
        'best.model'
    )


def _set_device(opt):
    if hasattr(opt,'gpu_id') and opt.gpu_id is not None:
        torch.cuda.set_device(opt.gpu_id)
        return torch.device(f'cuda:{opt.gpu_id}')
    return torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )


def _smart_load_state_dict(module:nn.Module,ckpt_path:str,device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'未找到权重文件: {ckpt_path}'
        )

    ckpt=torch.load(
        ckpt_path,
        map_location=device
    )

    if isinstance(ckpt,dict) and 'model' in ckpt:
        print(
            '[Checkpoint Info] '
            f'type={ckpt.get("type")} | '
            f'epoch={ckpt.get("epoch")} | '
            f'n_iter={ckpt.get("n_iter")} | '
            f'best_val_psnr={ckpt.get("best_val_psnr")} | '
            f'best_test_psnr={ckpt.get("best_test_psnr")}'
        )
        state=ckpt['model']
    else:
        print('[Checkpoint Info] old pure state_dict checkpoint')
        state=ckpt

    msg=module.load_state_dict(
        state,
        strict=False
    )

    print(
        f'[Load] 从 {ckpt_path} 加载完成：{msg}'
    )


def main():
    opt,MODEL_CLASS=_parse_all_args()

    device=_set_device(opt)
    print(f'Using device: {device}')

    model:TrainTask=MODEL_CLASS(opt)

    ckpt_path=opt.ckpt or _infer_default_best_path()
    print(f'权重路径: {ckpt_path}')

    if hasattr(model,'ema_model'):
        _smart_load_state_dict(
            model.ema_model,
            ckpt_path,
            device
        )
    else:
        _smart_load_state_dict(
            getattr(model,'model',model),
            ckpt_path,
            device
        )

    model.test(
        n_iter=opt.test_iter,
        epoch=opt.epoch_for_log
    )


if __name__=="__main__":
    main()