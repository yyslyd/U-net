import json

import pytest
import torch

from train import build_data_loaders
from utils.splits import create_split_manifest, load_split_manifest, save_split_manifest


SHA256 = "a" * 64


def test_split_is_deterministic_disjoint_and_exhaustive():
    ids = [f"tile_{index:02d}" for index in range(10)]

    first = create_split_manifest(ids, val_fraction=0.3, seed=42,
                                  checkpoint_filename="checkpoint_epoch1.pth",
                                  checkpoint_sha256=SHA256, mask_values=[0, 1],
                                  image_scale=0.5, bilinear=False)
    second = create_split_manifest(ids, val_fraction=0.3, seed=42,
                                   checkpoint_filename="checkpoint_epoch1.pth",
                                   checkpoint_sha256=SHA256, mask_values=[0, 1],
                                   image_scale=0.5, bilinear=False)

    assert first == second
    assert len(first["val_ids"]) == 3
    assert set(first["train_ids"]).isdisjoint(first["val_ids"])
    assert set(first["train_ids"]) | set(first["val_ids"]) == set(ids)
    assert first["dataset_ids"] == ids


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["val_ids"].append(data["train_ids"][0]), "disjoint"),
        (lambda data: data["train_ids"].pop(), "exhaustive"),
        (lambda data: data.update(schema_version=99), "schema"),
        (lambda data: data.update(checkpoint_filename="other.pth"), "checkpoint"),
    ],
)
def test_invalid_manifest_is_rejected(tmp_path, mutation, message):
    ids = ["a", "b", "c", "d"]
    manifest = create_split_manifest(ids, 0.25, 0, "checkpoint.pth", SHA256,
                                     [0, 1], 0.5, False)
    mutation(manifest)
    path = tmp_path / "checkpoint.split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_split_manifest(path, dataset_ids=ids, checkpoint_filename="checkpoint.pth",
                            checkpoint_sha256=SHA256)


def test_manifest_round_trip(tmp_path):
    ids = ["a", "b", "c", "d"]
    manifest = create_split_manifest(ids, 0.25, 7, "checkpoint.pth", SHA256,
                                     [0, 1], 0.5, False)
    path = tmp_path / "checkpoint.split.json"

    save_split_manifest(path, manifest)

    assert load_split_manifest(path, ids, "checkpoint.pth", SHA256) == manifest


def test_manifest_rejects_checkpoint_with_same_name_but_different_content(tmp_path):
    ids = ["a", "b"]
    manifest = create_split_manifest(ids, 0.5, 0, "checkpoint.pth", SHA256,
                                     [0, 1], 1.0, True)
    path = tmp_path / "checkpoint.split.json"
    save_split_manifest(path, manifest)

    with pytest.raises(ValueError, match="SHA-256"):
        load_split_manifest(path, ids, "checkpoint.pth", "b" * 64)


def test_validation_loader_keeps_incomplete_final_batch():
    dataset = torch.utils.data.TensorDataset(torch.arange(5))

    _, validation = build_data_loaders(dataset, [0, 1], [2, 3, 4], batch_size=2, seed=0)

    assert [batch[0].tolist() for batch in validation] == [[2, 3], [4]]
