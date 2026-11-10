from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """
    CelebA Blond_Hair classification with Male as the spurious attribute.

    y:
        Blond_Hair

    a:
        Male

    group:
        2 * y + a
    """

    SPLIT_MAP = {
        "train": 0,
        "val": 1,
        "valid": 1,
        "test": 2,
    }

    n_classes = 2

    def __init__(
        self,
        basedir,
        split,
        transform=None,
        indexes=None,
        seed=0,
    ):
        del seed

        self.basedir = Path(
            basedir
        )

        self.transform = (
            transform
        )

        split = str(
            split
        ).lower()

        if split not in self.SPLIT_MAP:
            raise ValueError(
                f"Unknown CelebA split: {split}"
            )

        split_id = self.SPLIT_MAP[
            split
        ]

        raw_root = (
            self.basedir
            / "raw"
        )

        if not raw_root.exists():
            raise FileNotFoundError(
                f"CelebA raw directory not found: {raw_root}"
            )

        # --------------------------------------------------
        # Find metadata CSVs.
        # --------------------------------------------------

        attr_matches = list(
            raw_root.rglob(
                "list_attr_celeba.csv"
            )
        )

        split_matches = list(
            raw_root.rglob(
                "list_eval_partition.csv"
            )
        )

        if not attr_matches:
            raise FileNotFoundError(
                "list_attr_celeba.csv not found "
                f"under {raw_root}"
            )

        if not split_matches:
            raise FileNotFoundError(
                "list_eval_partition.csv not found "
                f"under {raw_root}"
            )

        attr_path = attr_matches[
            0
        ]

        split_path = split_matches[
            0
        ]

        # --------------------------------------------------
        # Find aligned images.
        # --------------------------------------------------

        image_directory = None

        for candidate in raw_root.rglob(
            "img_align_celeba"
        ):
            if not candidate.is_dir():
                continue

            if any(
                candidate.glob(
                    "*.jpg"
                )
            ):
                image_directory = (
                    candidate
                )

                break

        if image_directory is None:
            raise FileNotFoundError(
                "Could not find CelebA aligned images "
                f"under {raw_root}"
            )

        self.image_directory = (
            image_directory
        )

        # --------------------------------------------------
        # Read Kaggle metadata.
        # --------------------------------------------------

        attributes = pd.read_csv(
            attr_path
        )

        partitions = pd.read_csv(
            split_path
        )

        if "image_id" not in attributes.columns:
            raise RuntimeError(
                "Expected image_id in "
                "list_attr_celeba.csv"
            )

        if "Blond_Hair" not in attributes.columns:
            raise RuntimeError(
                "Blond_Hair column missing."
            )

        if "Male" not in attributes.columns:
            raise RuntimeError(
                "Male column missing."
            )

        if "image_id" not in partitions.columns:
            raise RuntimeError(
                "Expected image_id in "
                "list_eval_partition.csv"
            )

        if "partition" not in partitions.columns:
            raise RuntimeError(
                "Expected partition column in "
                "list_eval_partition.csv"
            )

        metadata = attributes.merge(
            partitions[
                [
                    "image_id",
                    "partition",
                ]
            ],
            on="image_id",
            how="inner",
            validate="one_to_one",
        )

        metadata = metadata[
            metadata[
                "partition"
            ]
            == split_id
        ].reset_index(
            drop=True
        )

        # --------------------------------------------------
        # Convert {-1,+1} or {0,1} safely to {0,1}.
        # --------------------------------------------------

        def binary_column(
            column,
        ):
            values = metadata[
                column
            ].to_numpy()

            unique = set(
                np.unique(
                    values
                ).tolist()
            )

            if unique.issubset(
                {-1, 1}
            ):
                values = (
                    values
                    == 1
                ).astype(
                    np.int64
                )

            elif unique.issubset(
                {0, 1}
            ):
                values = values.astype(
                    np.int64
                )

            else:
                raise RuntimeError(
                    f"Unexpected values in {column}: "
                    f"{sorted(unique)}"
                )

            return values

        y = binary_column(
            "Blond_Hair"
        )

        a = binary_column(
            "Male"
        )

        # --------------------------------------------------
        # Optional deterministic subset support.
        # --------------------------------------------------

        if indexes is not None:
            indexes = np.asarray(
                indexes,
                dtype=np.int64,
            )

            metadata = metadata.iloc[
                indexes
            ].reset_index(
                drop=True
            )

            y = y[
                indexes
            ]

            a = a[
                indexes
            ]

        self.metadata = (
            metadata
        )

        self.y_array = (
            np.asarray(
                y,
                dtype=np.int64,
            )
        )

        self.confounder_array = (
            np.asarray(
                a,
                dtype=np.int64,
            )
        )

        self.group_array = (
            self.y_array
            * 2
            + self.confounder_array
        )

        self.filename_array = (
            self.metadata[
                "image_id"
            ]
            .astype(str)
            .to_numpy()
        )

        # Aliases useful in newer code.
        self.targets = torch.as_tensor(
            self.y_array
        )

        self.spurious = torch.as_tensor(
            self.confounder_array
        )

        self.groups = torch.as_tensor(
            self.group_array
        )

    def __len__(
        self,
    ):
        return len(
            self.y_array
        )

    def __getitem__(
        self,
        index,
    ):
        filename = (
            self.filename_array[
                index
            ]
        )

        image_path = (
            self.image_directory
            / filename
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        y = int(
            self.y_array[
                index
            ]
        )

        group = int(
            self.group_array[
                index
            ]
        )

        attribute = int(
            self.confounder_array[
                index
            ]
        )

        return (
            image,
            y,
            group,
            attribute,
        )
