import logging
import random
import sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from os.path import splitext
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.segmentation import encode_mask


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
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0,
                 mask_suffix: str = '', augment: bool = False, mask_values=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment

        image_files = self._files_by_id(self.images_dir, suffix='')
        mask_files = self._files_by_id(self.mask_dir, suffix=self.mask_suffix)
        if not image_files:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        missing_masks = sorted(set(image_files) - set(mask_files))
        missing_images = sorted(set(mask_files) - set(image_files))
        if missing_masks or missing_images:
            details = []
            if missing_masks:
                details.append(f'missing masks for {missing_masks}')
            if missing_images:
                details.append(f'missing images for {missing_images}')
            raise ValueError(f'Unmatched dataset files: {"; ".join(details)}')

        self.ids = sorted(image_files)

        # Pre-cache file paths
        self._img_paths = image_files
        self._mask_paths = mask_files

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        if mask_values is None:
            logging.info('Scanning mask files to determine unique values')
            unique = [unique_mask_values(id_, self.mask_dir, self.mask_suffix) for id_ in self.ids]
            if all(values.ndim == 1 for values in unique):
                self.mask_values = np.unique(np.concatenate(unique)).tolist()
            elif all(values.ndim == 2 for values in unique):
                self.mask_values = np.unique(np.concatenate(unique), axis=0).tolist()
            else:
                dimensions = sorted({values.ndim for values in unique})
                raise ValueError(f'Masks must consistently be scalar or color labels, found {dimensions}')
        else:
            self.mask_values = list(mask_values)
        logging.info(f'Unique mask values: {self.mask_values}')

        # Preload ALL data into RAM for zero-IO training
        logging.info('Preloading all data into RAM...')
        self._images = {}
        self._masks = {}
        for name in tqdm(self.ids, desc='Preloading', disable=not sys.stderr.isatty()):
            img = load_image(self._img_paths[name])
            mask = load_image(self._mask_paths[name])
            if img.size != mask.size:
                raise ValueError(
                    f'{name}: image and mask size differ '
                    f'({img.size[0]}x{img.size[1]} vs {mask.size[0]}x{mask.size[1]})'
                )

            # Preprocess image
            img_tensor = self._preprocess_img(img)
            # Preprocess mask
            mask_tensor = self._preprocess_mask(mask, source=str(self._mask_paths[name]))

            self._images[name] = img_tensor
            self._masks[name] = mask_tensor

        ram_mb = sum(t.nbytes for t in self._images.values()) + sum(t.nbytes for t in self._masks.values())
        logging.info(f'Preloaded {len(self.ids)} samples into RAM ({ram_mb / 1024**2:.0f} MB)')

    @staticmethod
    def _files_by_id(directory: Path, suffix: str):
        files = {}
        for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
            if not path.is_file() or path.name.startswith('.'):
                continue
            stem = path.stem
            if suffix:
                if not stem.endswith(suffix):
                    continue
                stem = stem[:-len(suffix)]
            if stem in files:
                raise ValueError(
                    f'Duplicate sample ID {stem!r} in {directory}: '
                    f'{files[stem].name!r} and {path.name!r}'
                )
            files[stem] = path
        return files

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

    def _preprocess_mask(self, pil_mask, source='mask'):
        w, h = pil_mask.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized masks would have no pixel'
        mask = encode_mask(np.asarray(pil_mask), self.mask_values, source=source)
        if self.scale != 1.0:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.int32)).resize((newW, newH), resample=Image.NEAREST),
                dtype=np.int64,
            )
        return torch.as_tensor(mask.copy()).long().contiguous()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        if is_mask:
            mask = encode_mask(np.asarray(pil_img), mask_values)
            if scale != 1.0:
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.int32)).resize((newW, newH), resample=Image.NEAREST),
                    dtype=np.int64,
                )
            return mask

        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)
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
    def __init__(self, images_dir, mask_dir, scale=1, augment=False, mask_values=None):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', augment=augment,
                         mask_values=mask_values)
