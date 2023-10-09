import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import numpy as np
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, Subset
import pandas as pd
import torch
import clip

'''
    23-03-05
    mmvid dataset used for DIFFUSION
'''

class MutilVox(Dataset):
    root = '/home/sunmeng/ab/ab5_dis/data/small_data/'

    def __init__(
            self, size=None,
            split="train",
            downscale_f=4,
            min_crop_f=0.5,
            max_crop_f=1.,
            random_crop=True,
    ):

        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        split_str = split

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "unseen_valid": 1,
            "unseen_test": 2,
            "all": None,
        }
        split = split_map[split]    # train - 0     23379条
        self.split = split

        fn = partial(os.path.join, self.root)

        splits = pd.read_csv(fn("small_data.csv"))     # 划分数据集的文件, train val test
        mask = slice(None) if split is None else (splits["split"] == split)

        pkl_file_prefix = "small"

        self.filename = splits[mask]["idx"].values      # 中的文件名

        cached_caption_pickle_fpath = os.path.join(
            self.root, f"{pkl_file_prefix}_{split_str}.pkl"       # 过滤之后的文本文件 train
        )

        with open(os.path.join(self.root, cached_caption_pickle_fpath), "rb") as f:
            self.cached_caption_lst = pickle.load(f)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.get_img(i))

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        # example["path"] = image     # clip处理

        min_side_len = min(image.shape[:2])

        crop_side_len = min_side_len

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        caption, caption_index = self.get_caption(
            i, deterministic=self.split != 0, return_index=True
        )

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["caption"] = caption

        return example

    def __len__(self):
        return len(self.filename)

    def get_caption(
            self,
            index,
            deterministic=False,
            return_index=False,
    ):
        captions = self.cached_caption_lst[index]

        if deterministic:
            caption_index = 0
        else:
            caption_index = torch.randint(low=0, high=len(captions), size=(1,))[
                0
            ].item()
        caption = captions[caption_index]
        if return_index:
            return caption, caption_index
        return caption

    def get_img(self, index):
        img_fpath = os.path.join(
            self.root,
            "img",
            f"{self.filename[index]}.png",
        )
        img = img_fpath
        return img

