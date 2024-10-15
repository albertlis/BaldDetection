import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from turbojpeg import TurboJPEG, TJPF_RGB


class CustomImageDataset(Dataset):
    def __init__(self, images_path: Path, annotations_path: Path, transform=None):
        self.images_path = images_path
        annotations = pd.read_csv(annotations_path)
        is_bald = annotations.Bald > 0
        self.bald_annotations = annotations[is_bald]
        self.other_annotations = annotations[~is_bald]
        self.transform = transform
        self.tjpeg = None


    def __len__(self) -> int:
        return 2 * len(self.bald_annotations)

    def __getitem__(self, idx) -> tuple[np.ndarray, float]:
        if idx < len(self.bald_annotations):
            annotations = self.bald_annotations
            is_bald = 1.
        else:
            annotations = self.other_annotations
            idx = random.randint(0, len(annotations))
            is_bald = 0.
        img_path = self.images_path / annotations.iloc[idx, 0]

        if self.tjpeg is None:
            self.tjpeg = TurboJPEG()

        with open(img_path, 'rb') as f:
            image = self.tjpeg.decode(f.read(), TJPF_RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, is_bald

def get_dataloaders(images_path: Path, annotations_path: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # TODO: add more here
        ToTensorV2(),
    ])

    dataset = CustomImageDataset(images_path, annotations_path, transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # TODO constant split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
