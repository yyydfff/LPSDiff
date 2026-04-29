import os
import os.path as osp
import re
from glob import glob
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


MAYO2016_ROOT = "mayo2016"


def natural_key(path):
    """Natural sort: 1.npy, 2.npy, 10.npy instead of 1.npy, 10.npy, 2.npy."""
    name = osp.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def resolve_pair_dirs(data_root, split):

    split_dir = osp.join(data_root, split)

    candidates = [
        ("LDCT", "NDCT"),
        ("ldct", "ndct"),
        ("quarter", "full"),
        ("Quarter", "Full"),
    ]

    for input_name, target_name in candidates:
        input_dir = osp.join(split_dir, input_name)
        target_dir = osp.join(split_dir, target_name)
        if osp.isdir(input_dir) and osp.isdir(target_dir):
            return input_dir, target_dir

    raise FileNotFoundError(
        f"Cannot find paired folders under {split_dir}. "
        f"Expected one of: {candidates}"
    )


def build_paired_npy_list(input_dir, target_dir):
    """Build matched LDCT/NDCT npy lists using the same filename."""
    input_files = sorted(glob(osp.join(input_dir, "*.npy")), key=natural_key)
    target_files_all = sorted(glob(osp.join(target_dir, "*.npy")), key=natural_key)

    print(f"Found {len(input_files)} input files and {len(target_files_all)} target files")

    if len(input_files) == 0 or len(target_files_all) == 0:
        raise ValueError(f"No input or target .npy files found in:\n{input_dir}\n{target_dir}")

    paired_inputs = []
    paired_targets = []
    missing = 0

    for input_file in input_files:
        target_file = osp.join(target_dir, osp.basename(input_file))
        if osp.exists(target_file):
            paired_inputs.append(input_file)
            paired_targets.append(target_file)
        else:
            missing += 1
            print(f"Warning: Missing target for {osp.basename(input_file)}")

    if len(paired_inputs) == 0:
        raise ValueError("No paired input-target files found. Please check file names.")

    print(f"After matching: {len(paired_inputs)} input, {len(paired_targets)} target, missing {missing}")
    return paired_inputs, paired_targets


def make_context_samples(input_files, target_files):

    min_length = min(len(input_files), len(target_files))
    input_files = input_files[:min_length]
    target_files = target_files[:min_length]

    if min_length < 3:
        raise ValueError("Need at least 3 paired slices for context mode")

    base_input = [
        [input_files[i - 1], input_files[i], input_files[i + 1]]
        for i in range(1, min_length - 1)
    ]
    base_target = target_files[1:-1]
    return base_input, base_target


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        self.mode = mode
        self.context = context
        print(f"Dataset: {dataset}, mode: {mode}, context: {context}")

        base_input = []
        base_target = []

        if dataset in ["mayo_2016_sim", "mayo_2016"]:

            data_root = os.environ.get("MAYO2016_ROOT", MAYO2016_ROOT)

            if mode not in ["train", "val", "test"]:
                raise ValueError(f"Unsupported mode for Mayo2016: {mode}. Use train / val / test.")

            input_dir, target_dir = resolve_pair_dirs(data_root, mode)
            input_files, target_files = build_paired_npy_list(input_dir, target_dir)

            if context:
                base_input, base_target = make_context_samples(input_files, target_files)
            else:
                min_length = min(len(input_files), len(target_files))
                base_input = input_files[:min_length]
                base_target = target_files[:min_length]

            self.input = base_input
            self.target = base_target
            print(f"Final dataset size - input: {len(self.input)}, target: {len(self.target)}")

        elif dataset == "mayo_2020":
            data_root = "./data_preprocess/gen_data/mayo_2020_npy"
            if dose == 10:
                patient_ids = ["C052", "C232", "C016", "C120", "C050"]
            elif dose == 25:
                patient_ids = ["L077", "L056", "L186", "L006", "L148"]
            else:
                raise ValueError(f"Unsupported Mayo2020 dose: {dose}")

            patient_lists = []
            for patient_id in patient_ids:
                patient_list = sorted(
                    glob(osp.join(data_root, patient_id + "_target_" + "*_img.npy")),
                    key=natural_key,
                )
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for patient_id in patient_ids:
                patient_list = sorted(
                    glob(osp.join(data_root, patient_id + f"_{dose}_" + "*_img.npy")),
                    key=natural_key,
                )
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        cat_patient_list.append([patient_list[i - 1], patient_list[i], patient_list[i + 1]])
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_input = patient_lists

        elif dataset == "piglet":
            data_root = "./data_preprocess/gen_data/piglet_npy"

            patient_list = sorted(glob(osp.join(data_root, "piglet_target_" + "*_img.npy")), key=natural_key)
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, f"piglet_{dose}_" + "*_img.npy")), key=natural_key)
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    cat_patient_list.append([patient_list[i - 1], patient_list[i], patient_list[i + 1]])
                base_input = cat_patient_list
            else:
                base_input = patient_list[1:len(patient_list) - 1]

        elif dataset == "phantom":
            data_root = "./data_preprocess/gen_data/xnat_npy"

            patient_list = sorted(glob(osp.join(data_root, "xnat_target" + "*_img.npy")), key=natural_key)[9:21]
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(
                glob(osp.join(data_root, "xnat_{:0>3d}_".format(dose) + "*_img.npy")),
                key=natural_key,
            )[9:21]
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    cat_patient_list.append([patient_list[i - 1], patient_list[i], patient_list[i + 1]])
                base_input = cat_patient_list
            else:
                base_input = patient_list[1:len(patient_list) - 1]

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.input = base_input
        self.target = base_target
        print(f"Input samples: {len(self.input)}")
        print(f"Target samples: {len(self.target)}")

    def __getitem__(self, index):
        input_path, target_path = self.input[index], self.target[index]

        if self.context:
            if isinstance(input_path, (list, tuple)):
                input_paths = list(input_path)
            else:
                # Compatible with old '~'-joined style. Remove possible empty prefix.
                input_paths = [p for p in str(input_path).split("~") if p]

            if len(input_paths) != 3:
                raise ValueError(f"Context mode expects 3 slices, got {len(input_paths)}: {input_paths}")

            inputs = [np.load(p)[np.newaxis, ...].astype(np.float32) for p in input_paths]
            input_arr = np.concatenate(inputs, axis=0)  # (3, H, W)
        else:
            input_arr = np.load(input_path)[np.newaxis, ...].astype(np.float32)  # (1, H, W)

        target_arr = np.load(target_path)[np.newaxis, ...].astype(np.float32)  # (1, H, W)

        input_arr = self.normalize_(input_arr)
        target_arr = self.normalize_(target_arr)
        return input_arr, target_arr

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):

        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return np.clip(img, 0, 1)



dataset_dict = {
    "train": partial(CTDataset, dataset="mayo_2016", mode="train", test_id=9, dose=25, context=True),
    "val": partial(CTDataset, dataset="mayo_2016", mode="val", test_id=9, dose=25, context=True),
    "mayo_2016_sim": partial(CTDataset, dataset="mayo_2016_sim", mode="test", test_id=9, dose=5, context=True),
    "mayo_2016": partial(CTDataset, dataset="mayo_2016", mode="test", test_id=9, dose=25, context=True),
    "mayo_2020": partial(CTDataset, dataset="mayo_2020", mode="test", test_id=None, dose=None, context=True),
    "piglet": partial(CTDataset, dataset="piglet", mode="test", test_id=None, dose=None, context=True),
    "phantom": partial(CTDataset, dataset="phantom", mode="test", test_id=None, dose=108, context=True),
}
