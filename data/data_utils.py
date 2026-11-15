from config import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler

import numpy as np
import pickle
import os
import glob
import hashlib
import json
import torch

class PredDataset(Dataset):
    def __init__(
        self,
        dataset,
        pred_labels,

    ):
        self.n_classes = dataset.n_classes
       
        self.dataset = dataset
        self.pred_labels = pred_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset[idx]
        pred_label = self.pred_labels[idx]
        return *items, pred_label
    

class MisDataset(Dataset):
    def __init__(
        self,
        dataset,
        mis_labels,
        indexes,

    ):
        self.n_classes = dataset.n_classes
       
        self.dataset = dataset
        self.indexes = indexes
        self.mis_labels = mis_labels

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        items = self.dataset[self.indexes[idx]]
        mis_label = self.mis_labels[idx]
        return *items, mis_label


class JointDataset(Dataset):
    def __init__(
        self,
        datasets,
    ):
        
        self.datasets = datasets
        self.n_datasets = len(datasets)

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        assert len(idx) == self.n_datasets, "wrong indexes"
        output = []
        for n,i in enumerate(idx):
            output.extend(self.datasets[n][i])
        return output

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

class SubsetDataset(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


def get_split_indexes(dataset : Dataset, ratio : float):
    """
    Get the indexes of the first and second subsets of the dataset. 
    The ratio specifies the first subset's proportion of the dataset.

    :param dataset: target dataset
    :param ratio: split ratio

    :return indexes_first_subset, indexes_second_subset
    """
    assert ratio > 0 and ratio < 1, "ratio is not within (0,1)"
    num = len(dataset)
    indexes = np.arange(num)
    np.random.shuffle(indexes)
    num1 = int(num*ratio)
    return indexes[0:num1], indexes[num1:]

# ------------------------------------------------------------
# Saved row partition (--split_file)
# ------------------------------------------------------------
# A partition fixes, once, which rows of the official train and val splits
# go to subset1 and subset2. Positions are rows of each official split in
# metadata.csv order (the same positions BiasedDataset uses with sel_indexes).
# Every stage (ERM, rater, final classifier) reads the same file, so the rows
# never depend on the global RNG, the seed of a later stage or a pickle left in
# a local dataset folder. Group labels are never used to build it.

PARTITION_FORMAT = "row-partition-v1"


def _rows_sha256(filenames):
    joined = "\n".join(str(name) for name in filenames)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _official_split_filenames(basedir, split, dataset=None):
    import pandas as pd
    metadata_path = os.path.join(basedir, "metadata.csv")
    if dataset is not None and not os.path.exists(metadata_path):
        # Datasets without a metadata.csv (CelebA): the rows in the order the
        # dataset class itself uses, so partition positions index the same rows.
        from .register import dataset_dict
        return np.asarray(dataset_dict[dataset](basedir, split).filename_array)
    metadata = pd.read_csv(metadata_path)
    split_i = ["train", "val", "test"].index(split)
    return metadata[metadata["split"] == split_i]["img_filename"].values


def make_partition(basedir, seed, split_train, split_val, dataset=None):
    """Fixed partition of the official train and val rows (seeded, label-blind).

    subset1 gets int(n * ratio) rows, subset2 the rest; both are stored sorted.
    dataset is only needed for datasets without a metadata.csv (e.g. "celeba").
    """
    partition = {
        "format": PARTITION_FORMAT,
        "seed": int(seed),
        "split_train": float(split_train),
        "split_val": float(split_val),
        "splits": {},
    }
    for split, ratio in (("train", split_train), ("val", split_val)):
        if not 0.0 < float(ratio) < 1.0:
            raise ValueError(f"split_{split} must be in (0, 1) to build a partition, got {ratio}")
        filenames = _official_split_filenames(basedir, split, dataset)
        n = len(filenames)
        order = np.random.RandomState(int(seed)).permutation(n)
        k = int(n * float(ratio))
        first, second = np.sort(order[:k]), np.sort(order[k:])
        partition["splits"][split] = {
            "n_rows": int(n),
            "rows_sha256": _rows_sha256(filenames),
            "indexes1": [int(i) for i in first],
            "indexes2": [int(i) for i in second],
            "indexes1_sha256": _rows_sha256(filenames[first]),
            "indexes2_sha256": _rows_sha256(filenames[second]),
        }
    validate_partition(partition)
    return partition


def validate_partition(partition):
    """Each split: subset1 and subset2 disjoint, together every row exactly once."""
    if partition.get("format") != PARTITION_FORMAT:
        raise ValueError(f"Unknown partition format: {partition.get('format')!r}")
    for split, key in (("train", "split_train"), ("val", "split_val")):
        part = partition["splits"][split]
        n = int(part["n_rows"])
        first, second = part["indexes1"], part["indexes2"]
        if len(first) != int(n * float(partition[key])) or len(first) + len(second) != n:
            raise ValueError(f"{split}: subset sizes {len(first)}/{len(second)} do not match "
                             f"{key}={partition[key]} of {n} rows")
        if sorted(first + second) != list(range(n)):
            raise ValueError(f"{split}: subsets overlap or do not cover all {n} rows")
    return partition


def partition_sha256(partition):
    """Identity of the rows in a partition (used in embedding-cache metadata)."""
    core = {
        "seed": partition["seed"],
        "split_train": partition["split_train"],
        "split_val": partition["split_val"],
        "splits": {split: {key: part[key] for key in ("n_rows", "rows_sha256", "indexes1", "indexes2")}
                   for split, part in partition["splits"].items()},
    }
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()


def save_partition(path, partition):
    """Write a partition once; an existing file must hold the same rows."""
    validate_partition(partition)
    if os.path.exists(path):
        existing = load_partition(path)
        if partition_sha256(existing) != partition_sha256(partition):
            raise FileExistsError(f"{path} already holds a different partition; it belongs to this ERM, "
                                  f"do not replace it.")
        return path
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(partition, f, indent=1)
    os.replace(tmp, path)
    return path


def load_partition(path):
    with open(path) as f:
        return validate_partition(json.load(f))


def partition_indexes(partition, split, ratio):
    """subset1/subset2 rows of an official split; the requested ratio must match the file."""
    key = f"split_{split}"
    if abs(float(ratio) - float(partition[key])) > 1e-9:
        raise ValueError(f"--{key} {ratio} does not match the saved partition ({partition[key]})")
    part = partition["splits"][split]
    return np.asarray(part["indexes1"], dtype=np.int64), np.asarray(part["indexes2"], dtype=np.int64)


def check_partition_rows(partition, split, filenames):
    """Abort if the official split's rows are not the rows the partition was built on."""
    part = partition["splits"][split]
    if len(filenames) != int(part["n_rows"]) or _rows_sha256(filenames) != part["rows_sha256"]:
        raise ValueError(f"The {split} rows of this dataset do not match the saved partition "
                         f"(metadata.csv differs from the one it was built on).")


def prepare_data(config):
    from .register import dataset_dict, transform_funcs
    dataset = config.dataset
    if dataset not in dataset_dict:
        raise ValueError(f"{dataset} not defined/registered")
    if dataset not in dataset_paths:
        raise ValueError(f"path to {dataset} not provided. Please add in config.py")
    
    dataset_class = dataset_dict[dataset]
    basedir = dataset_paths[dataset]
    if dataset in transform_funcs:
        eval_transform = transform_funcs[dataset](config.resolution, False, False)
        train_transform = transform_funcs[dataset](config.resolution, True, True)
    else:
        train_transform = None
        eval_transform = None

    train_dataset = dataset_class(basedir, "train", train_transform)
    train_no_aug_dataset = dataset_class(basedir, "train", eval_transform)
    val_dataset = dataset_class(basedir, "val", eval_transform)

    test_dataset = dataset_class(basedir, "test", eval_transform)
    datasets =  {
            "train": train_dataset,
            "train_no_aug": train_no_aug_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

    split_file = getattr(config, "split_file", "") or ""
    partition = None
    if split_file:
        partition = load_partition(split_file)
        if config.split_train >= 1.0:
            raise ValueError("--split_file is for held-out runs: set --split_train to its ratio "
                             f"({partition['split_train']}).")
        if not hasattr(train_dataset, "filename_array"):
            raise ValueError("--split_file supports image datasets with metadata.csv only.")
        check_partition_rows(partition, "train", train_dataset.filename_array)
        check_partition_rows(partition, "val", val_dataset.filename_array)


    if config.split_train < 1.0:
        train_split_path = os.path.join(basedir, f"train_split_{config.split_train:.2f}_seed{config.seed}.pickle")
        if partition is not None:
            train_indexes1, train_indexes2 = partition_indexes(partition, "train", config.split_train)
        elif os.path.exists(train_split_path):
            with open(train_split_path, "rb") as f:
                train_indexes1, train_indexes2 = pickle.load(f)
        else:
            train_indexes1, train_indexes2 = get_split_indexes(train_dataset, config.split_train)
            with open(train_split_path, "wb") as f:
                pickle.dump((train_indexes1, train_indexes2), f)
        train_subset1 = dataset_class(basedir, "train", train_transform, train_indexes1, config.seed)
        train_subset2 = dataset_class(basedir, "train", train_transform, train_indexes2, config.seed)

        train_no_aug_subset1 = dataset_class(basedir, "train", eval_transform, train_indexes1, config.seed)
        train_no_aug_subset2 = dataset_class(basedir, "train", eval_transform, train_indexes2, config.seed)
        datasets.update({"train_subset1": train_subset1, 
                        "train_subset2": train_subset2, 
                        "train_no_aug_subset1": train_no_aug_subset1, 
                        "train_no_aug_subset2": train_no_aug_subset2,
                        "train_indexes1": train_indexes1,
                        "train_indexes2": train_indexes2,
                        })
    if config.split_val < 1.0:
        val_split_path = os.path.join(basedir, f"val_split_{config.split_val:.2f}_seed{config.seed}.pickle")
        if partition is not None:
            val_indexes1, val_indexes2 = partition_indexes(partition, "val", config.split_val)
        elif os.path.exists(val_split_path):
            with open(val_split_path, "rb") as f:
                val_indexes1, val_indexes2 = pickle.load(f)
        else:
            val_indexes1, val_indexes2 = get_split_indexes(val_dataset, config.split_val)
            with open(val_split_path, "wb") as f:
                pickle.dump((val_indexes1, val_indexes2), f)

        val_subset1 = dataset_class(basedir, "val", eval_transform, val_indexes1, config.seed)
        val_subset2 = dataset_class(basedir, "val", eval_transform, val_indexes2, config.seed)
        if "no_aug" in config.train_split:
            train_val_dataset = dataset_class(basedir, "train", eval_transform)
        else:
            train_val_dataset = dataset_class(basedir, "train", train_transform)
        train_val_dataset.y_array = np.concatenate([train_val_dataset.y_array, val_subset1.y_array])
        train_val_dataset.group_array = np.concatenate([train_val_dataset.group_array, val_subset1.group_array])
        train_val_dataset.confounder_array = np.concatenate([train_val_dataset.confounder_array, val_subset1.confounder_array])
        if config.dataset in ["civilcomments", "multinli"]:
            train_val_dataset.data = torch.cat([train_val_dataset.data, val_subset1.data])
        else:
            train_val_dataset.filename_array = np.concatenate([train_val_dataset.filename_array, val_subset1.filename_array])
        
        datasets.update({"val_subset1": val_subset1, 
                        "val_subset2": val_subset2, 
                        "val_indexes1": val_indexes1,
                        "val_indexes2": val_indexes2,
                        })
        if "no_aug" in config.train_split:
             datasets.update({"train_val_no_aug": train_val_dataset})
        else:
            datasets.update({"train_val": train_val_dataset})
    dataloaders = {}
    for k in datasets:
        if "indexes" in k:
            continue
        if "train" in k and "no_aug" not in k:
            loader = DataLoader(
                datasets[k],
                batch_size=config.batch_size,
                pin_memory=True,
                shuffle=True,
                num_workers=config.num_workers,
            )
        else:
            loader = DataLoader(
                datasets[k],
                batch_size=config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        dataloaders[k] = loader
    
    return datasets, dataloaders


