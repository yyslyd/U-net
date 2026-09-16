"""Deterministic train/validation splits and checkpoint sidecar manifests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch


SPLIT_MANIFEST_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_file_records(dataset) -> list[dict[str, str]]:
    """Hash each paired input so a manifest detects content changes, not only ID changes."""
    return [
        {
            "id": sample_id,
            "image_file": dataset._img_paths[sample_id].name,
            "image_sha256": file_sha256(dataset._img_paths[sample_id]),
            "mask_file": dataset._mask_paths[sample_id].name,
            "mask_sha256": file_sha256(dataset._mask_paths[sample_id]),
        }
        for sample_id in dataset.ids
    ]


def create_split_ids(dataset_ids: Iterable[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = list(dataset_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")
    if not 0 <= val_fraction <= 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    permutation = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(len(ids) * val_fraction)
    n_train = len(ids) - n_val
    return ([ids[index] for index in permutation[:n_train]],
            [ids[index] for index in permutation[n_train:]])


def create_split_manifest(
    dataset_ids: Iterable[str],
    val_fraction: float,
    seed: int,
    checkpoint_filename: str,
    checkpoint_sha256: str,
    mask_values: Iterable,
    image_scale: float,
    bilinear: bool,
    dataset_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ids = list(dataset_ids)
    train_ids, val_ids = create_split_ids(ids, val_fraction, seed)
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "validation_fraction": val_fraction,
        "dataset_ids": ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "mask_values": list(mask_values),
        "image_scale": image_scale,
        "bilinear": bilinear,
    }
    if dataset_files is not None:
        manifest["dataset_files"] = dataset_files
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(
    manifest: dict[str, Any],
    dataset_ids: Iterable[str] | None = None,
    checkpoint_filename: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_files: list[dict[str, str]] | None = None,
) -> None:
    required = {
        "schema_version", "checkpoint_filename", "checkpoint_sha256", "seed",
        "validation_fraction", "dataset_ids", "train_ids", "val_ids",
        "mask_values", "image_scale", "bilinear",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"split manifest missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported split manifest schema: {manifest['schema_version']}")
    if not _SHA256_RE.fullmatch(str(manifest["checkpoint_sha256"])):
        raise ValueError("split manifest checkpoint SHA-256 is invalid")
    if not isinstance(manifest["seed"], int):
        raise ValueError("split manifest seed must be an integer")
    if not 0 <= manifest["validation_fraction"] <= 1:
        raise ValueError("split manifest validation fraction must be between 0 and 1")
    if not 0 < manifest["image_scale"] <= 1:
        raise ValueError("split manifest image_scale must be in (0, 1]")
    if not isinstance(manifest["bilinear"], bool):
        raise ValueError("split manifest bilinear must be boolean")
    if not manifest["mask_values"]:
        raise ValueError("split manifest mask_values must not be empty")

    all_ids = manifest["dataset_ids"]
    train_ids = manifest["train_ids"]
    val_ids = manifest["val_ids"]
    if any(not isinstance(item, str) for item in all_ids + train_ids + val_ids):
        raise ValueError("split manifest IDs must be strings")
    if len(all_ids) != len(set(all_ids)) or len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("split manifest IDs must not contain duplicates")
    if set(train_ids) & set(val_ids):
        raise ValueError("split manifest train and validation IDs must be disjoint")
    if set(train_ids) | set(val_ids) != set(all_ids):
        raise ValueError("split manifest train and validation IDs must be exhaustive")

    if dataset_ids is not None and list(dataset_ids) != all_ids:
        raise ValueError("split manifest dataset IDs/order do not match the current dataset")
    if checkpoint_filename is not None and manifest["checkpoint_filename"] != checkpoint_filename:
        raise ValueError("split manifest checkpoint filename does not match")
    if checkpoint_sha256 is not None and manifest["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("split manifest checkpoint SHA-256 does not match")
    if dataset_files is not None:
        if "dataset_files" not in manifest:
            raise ValueError("split manifest has no dataset file hashes")
        if manifest["dataset_files"] != dataset_files:
            raise ValueError("split manifest dataset file hashes do not match the current dataset")


def save_split_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    validate_split_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_split_manifest(
    path: Path | str,
    dataset_ids: Iterable[str] | None = None,
    checkpoint_filename: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Checkpoint-associated split manifest not found: {source}. "
            "Legacy checkpoints cannot be evaluated reproducibly without a saved split manifest."
        )
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid split manifest {source}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("split manifest root must be a JSON object")
    validate_split_manifest(manifest, dataset_ids, checkpoint_filename, checkpoint_sha256, dataset_files)
    return manifest
