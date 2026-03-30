import logging
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Pre-cache file paths
        self._img_paths = {}
        self._mask_paths = {}
        for name in self.ids:
            img_matches = list(self.images_dir.glob(name + '.*'))
            mask_matches = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
            if img_matches and mask_matches:
                self._img_paths[name] = img_matches[0]
                self._mask_paths[name] = mask_matches[0]

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        if os.name == 'nt':
            unique = []
            for id_ in tqdm(self.ids, desc='Scanning masks'):
                unique.append(unique_mask_values(id_, self.mask_dir, self.mask_suffix))
        else:
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        # Build LUT for fast mask remapping
        self._mask_lut = None
        if all(isinstance(v, (int, float)) for v in self.mask_values):
            max_val = max(int(v) for v in self.mask_values)
            if max_val < 256:
                self._mask_lut = np.full(256, 0, dtype=np.int64)
                for i, v in enumerate(self.mask_values):
                    self._mask_lut[int(v)] = i

        # Preload ALL data into RAM for zero-IO training
        logging.info('Preloading all data into RAM...')
        self._images = {}
        self._masks = {}
        for name in tqdm(self.ids, desc='Preloading'):
            img = load_image(self._img_paths[name])
            mask = load_image(self._mask_paths[name])

            # Preprocess image
            img_tensor = self._preprocess_img(img)
            # Preprocess mask
            mask_tensor = self._preprocess_mask(mask)

            self._images[name] = img_tensor
            self._masks[name] = mask_tensor

        ram_mb = sum(t.nbytes for t in self._images.values()) + sum(t.nbytes for t in self._masks.values())
        logging.info(f'Preloaded {len(self.ids)} samples into RAM ({ram_mb / 1024**2:.0f} MB)')

    def _preprocess_img(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        if self.scale != 1.0:
            pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return torch.as_tensor(img.copy()).float().contiguous()

    def _preprocess_mask(self, pil_mask):
        w, h = pil_mask.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        if self.scale != 1.0:
            pil_mask = pil_mask.resize((newW, newH), resample=Image.NEAREST)
        mask = np.asarray(pil_mask)
        if self._mask_lut is not None:
            mask = self._mask_lut[mask]
        else:
            result = np.zeros(mask.shape[:2], dtype=np.int64)
            for i, v in enumerate(self.mask_values):
                if mask.ndim == 2:
                    result[mask == v] = i
                else:
                    result[(mask == v).all(-1)] = i
            mask = result
        return torch.as_tensor(mask.copy()).long().contiguous()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = self._images[name]
        mask = self._masks[name]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
            # Random vertical flip
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, [1, 2])
                mask = torch.rot90(mask, k, [0, 1])
            # Color jitter (image only)
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
            if random.random() > 0.5:
                img = TF.adjust_contrast(img, 0.8 + random.random() * 0.4)

        return {
            'image': img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, augment=False):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', augment=augment)
