import argparse
import torch

from models.basic_template import TrainTask
from models import model_dict


def _parse_all_args():
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()

    if default_opt.model_name not in model_dict:
        raise ValueError(
            f"Unknown model_name: {default_opt.model_name}. Available: {list(model_dict.keys())}"
        )

    MODEL_CLASS = model_dict[default_opt.model_name]

    private_parser = (
        MODEL_CLASS.build_options()
        if hasattr(MODEL_CLASS, "build_options")
        else argparse.ArgumentParser()
    )

    opt = private_parser.parse_args(
        unknown_opt,
        namespace=default_opt
    )

    return opt, MODEL_CLASS


def _set_device(opt):
    if hasattr(opt, "gpu_id") and opt.gpu_id is not None:
        torch.cuda.set_device(opt.gpu_id)
        return torch.device(f"cuda:{opt.gpu_id}")

    return torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def main():
    opt, MODEL_CLASS = _parse_all_args()

    device = _set_device(opt)
    print(f"Using device: {device}")

    model: TrainTask = MODEL_CLASS(opt)

    model.fit()


if __name__ == "__main__":
    main()
