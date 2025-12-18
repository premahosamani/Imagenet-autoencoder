#!/usr/bin/env python

import os
import torch
import torch.utils.data as data
import rasterio
import numpy as np


class ImageDataset(data.Dataset):
    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.init()

    def init(self):
        self.im_names = []
        self.targets = []

        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split(' ')
                self.im_names.append(path)
                self.targets.append(int(label))

    def __getitem__(self, index):
        im_name = self.im_names[index]

        # Read 5-channel TIFF
        with rasterio.open(im_name) as src:
            img = src.read()              # shape: (5, H, W)

        img = img.astype(np.float32)

        # Simple normalization (safe for multispectral)
        max_val = img.max()
        if max_val > 0:
            img = img / max_val

        img = torch.from_numpy(img)       # (5, H, W)

        # Autoencoder: input == target
        return img, img

    def __len__(self):
        return len(self.im_names)


def train_loader(args):
    train_dataset = ImageDataset(args.train_list)

    if args.parallel == 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            rank=args.rank,
            num_replicas=args.world_size,
            shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    return train_loader


def val_loader(args):
    val_dataset = ImageDataset(args.val_list)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return val_loader
