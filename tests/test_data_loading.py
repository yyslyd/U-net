from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.data_loading import BasicDataset
from utils.segmentation import AMTOWN02_MASK_VALUES


def _write_rgb(path: Path, value: int = 0) -> None:
    Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, values) -> None:
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def _dataset(tmp_path: Path, names=("zeta", "alpha"), mask_values=(0, 1)):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index, name in enumerate(names):
        _write_rgb(images / f"{name}.png", index)
        _write_mask(masks / f"{name}.png", [[0, 1], [1, 0]])
    return BasicDataset(images, masks, mask_values=mask_values)


def test_dataset_ids_are_sorted_independent_of_directory_iteration(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(reversed(list(self.glob("*")))))

    reordered = BasicDataset(dataset.images_dir, dataset.mask_dir, mask_values=(0, 1))

    assert dataset.ids == reordered.ids == ["alpha", "zeta"]


@pytest.mark.parametrize("missing_side", ["mask", "image"])
def test_dataset_rejects_missing_pairs(tmp_path, missing_side):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_rgb(images / "paired.png")
    _write_mask(masks / "paired.png", [[0]])
    if missing_side == "mask":
        _write_rgb(images / "orphan.png")
    else:
        _write_mask(masks / "orphan.png", [[0]])

    with pytest.raises(ValueError, match="Unmatched dataset files"):
        BasicDataset(images, masks, mask_values=(0, 1))


@pytest.mark.parametrize("directory", ["images", "masks"])
def test_dataset_rejects_duplicate_stems(tmp_path, directory):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_rgb(images / "sample.png")
    _write_mask(masks / "sample.png", [[0]])
    if directory == "images":
        _write_rgb(images / "sample.jpg")
    else:
        _write_mask(masks / "sample.bmp", [[0]])

    with pytest.raises(ValueError, match="Duplicate"):
        BasicDataset(images, masks, mask_values=(0, 1))


def test_amtown02_labels_are_encoded_in_declared_order(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((1, 15, 3), dtype=np.uint8)).save(images / "all.png")
    _write_mask(masks / "all.png", [AMTOWN02_MASK_VALUES])

    dataset = BasicDataset(images, masks, mask_values=AMTOWN02_MASK_VALUES)

    assert dataset.mask_values == list(AMTOWN02_MASK_VALUES)
    assert dataset[0]["mask"].tolist() == [list(range(15))]


def test_dataset_rejects_unknown_mask_label_instead_of_mapping_to_background(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((1, 2, 3), dtype=np.uint8)).save(images / "bad.png")
    _write_mask(masks / "bad.png", [[0, 99]])

    with pytest.raises(ValueError, match=r"bad.*99"):
        BasicDataset(images, masks, mask_values=AMTOWN02_MASK_VALUES)


def test_dataset_rejects_image_mask_size_mismatch(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(images / "bad.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks / "bad.png")

    with pytest.raises(ValueError, match=r"bad.*size"):
        BasicDataset(images, masks, mask_values=(0, 1))


def test_dataset_validates_raw_labels_before_downscaling(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(images / "rare.png")
    Image.fromarray(np.array([[99, 0], [0, 0]], dtype=np.uint8)).save(masks / "rare.png")

    with pytest.raises(ValueError, match=r"rare.*99"):
        BasicDataset(images, masks, scale=0.5, mask_values=(0, 1))


def test_static_preprocess_validates_raw_labels_before_downscaling():
    mask = Image.fromarray(np.array([[99, 0], [0, 0]], dtype=np.uint8))

    with pytest.raises(ValueError, match="99"):
        BasicDataset.preprocess((0, 1), mask, scale=0.5, is_mask=True)
