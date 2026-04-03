"""Class-conditional 3DGS dataset wrapper.

Wraps Standard3DGenDataset to pair each sample with its class label,
filter out invalid classes, and optionally select a subset of feature channels.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloaders.standard_3dgen_loader import (
    Standard3DGenDataset,
    _normalize_point_cloud_numpy,
    extract_directory_info,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)

FULL_3DGS_FEATURE_DIM = 59
DC_ONLY_FEATURE_INDICES = (0, 1, 2, 3, 4, 20, 36, 52, 53, 54, 55, 56, 57, 58)
PRELOAD_CACHE_VERSION = 7
LAZY_CACHE_LOCK_STRIPES = 256
_PRELOAD_WORKER_STATE = {}


def _default_preload_cache_root() -> Path:
    """Return the RAM-backed cache directory used for shared CPU preload."""
    candidate = Path("/dev/shm") / "3dgen_preload_cache"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "preload_to_cpu requires a writable /dev/shm so the shared cache stays in RAM only"
        ) from exc
    if not os.access(candidate, os.W_OK):
        raise RuntimeError(
            "preload_to_cpu requires a writable /dev/shm so the shared cache stays in RAM only"
        )
    return candidate


def _hash_strings(hasher: "hashlib._Hash", values) -> None:
    for value in values:
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)


def _hash_array(hasher: "hashlib._Hash", array) -> None:
    arr = np.ascontiguousarray(np.asarray(array))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.view(np.uint8).tobytes())


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return np.dtype(torch.empty((), dtype=dtype).numpy().dtype)


def _torch_dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported cache dtype: {dtype}")


def _dtype_name_to_torch(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported cache dtype name: {name}")


def _element_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _open_tensor_memmap(
    path: str,
    mode: str,
    tensor_dtype: torch.dtype,
    shape: tuple[int, ...],
) -> tuple[np.memmap, torch.Tensor]:
    if tensor_dtype == torch.bfloat16:
        backing = np.memmap(path, mode=mode, dtype=np.uint16, shape=shape)
        tensor = torch.from_numpy(backing).view(torch.bfloat16)
    else:
        backing = np.memmap(path, mode=mode, dtype=_torch_dtype_to_numpy(tensor_dtype), shape=shape)
        tensor = torch.from_numpy(backing)
    return backing, tensor


def _load_preload_point_cloud(
    base_dataset: Standard3DGenDataset,
    real_idx: int,
) -> np.ndarray:
    """Load and normalize one plane-grid point cloud via Standard3DGenDataset."""
    hash_key = base_dataset.keys[real_idx]
    tar_gz_path = base_dataset.obj_data[hash_key]
    directory_number, filename = extract_directory_info(tar_gz_path)
    point_cloud, _ = base_dataset._load_3dgs_data(directory_number, filename)

    if base_dataset.mean is not None and base_dataset.std is not None:
        point_cloud = _normalize_point_cloud_numpy(point_cloud, base_dataset.mean, base_dataset.std)

    return point_cloud.astype(np.float32, copy=False)


def _select_point_cloud_features_numpy(
    point_cloud: np.ndarray,
    feature_indices: Optional[np.ndarray],
) -> np.ndarray:
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if feature_indices is None:
        return point_cloud
    if point_cloud.ndim == 2:
        return point_cloud[:, feature_indices]
    if point_cloud.ndim == 3:
        return point_cloud[feature_indices]
    raise ValueError(f"Expected point_cloud with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")


def _plane_point_cloud_to_grid_numpy(point_cloud: np.ndarray) -> np.ndarray:
    """Normalize plane-based numpy point clouds to (D, H, W)."""
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if point_cloud.ndim == 3:
        return np.ascontiguousarray(point_cloud)
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected point_cloud with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")
    n, d = point_cloud.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"N={n} is not a perfect square")
    return np.ascontiguousarray(point_cloud.reshape(side, side, d).transpose(2, 0, 1))


def _build_preload_grids(
    base_dataset: Standard3DGenDataset,
    real_idx: int,
    feature_indices: Optional[np.ndarray],
    return_full_for_render: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Build plane-grid arrays for the shared preload cache."""
    pc_full = _load_preload_point_cloud(base_dataset, real_idx)
    pc = _select_point_cloud_features_numpy(pc_full, feature_indices)
    pc_plane = _plane_point_cloud_to_grid_numpy(pc)
    if return_full_for_render:
        return pc_plane, _plane_point_cloud_to_grid_numpy(pc_full)
    return pc_plane, None


def _init_preload_worker(
    base_dataset: Standard3DGenDataset,
    feature_indices: Optional[np.ndarray],
    return_full_for_render: bool,
    pc_path: str,
    pc_shape: tuple[int, ...],
    pc_dtype_name: str,
    labels_path: str,
    labels_shape: tuple[int, ...],
    labels_dtype: str,
    pc_full_path: Optional[str],
    pc_full_shape: Optional[tuple[int, ...]],
    pc_full_dtype_name: Optional[str],
) -> None:
    """Initialize per-process state for parallel preload workers."""
    global _PRELOAD_WORKER_STATE
    _PRELOAD_WORKER_STATE = {
        "base_dataset": base_dataset,
        "feature_indices": feature_indices,
        "return_full_for_render": return_full_for_render,
        "pc_tensor": _open_tensor_memmap(
            pc_path,
            mode="r+",
            tensor_dtype=_dtype_name_to_torch(pc_dtype_name),
            shape=tuple(pc_shape),
        )[1],
        "labels_memmap": np.memmap(
            labels_path, mode="r+", dtype=np.dtype(labels_dtype), shape=tuple(labels_shape)
        ),
        "pc_full_tensor": None,
    }
    if return_full_for_render:
        _PRELOAD_WORKER_STATE["pc_full_tensor"] = _open_tensor_memmap(
            pc_full_path,
            mode="r+",
            tensor_dtype=_dtype_name_to_torch(pc_full_dtype_name),
            shape=tuple(pc_full_shape),
        )[1]


def _preload_worker_write_sample(task: tuple[int, int, int]) -> int:
    """Write one sample directly into the shared preload memmaps."""
    slot, real_idx, label = task
    state = _PRELOAD_WORKER_STATE
    pc_plane, pc_full_plane = _build_preload_grids(
        base_dataset=state["base_dataset"],
        real_idx=real_idx,
        feature_indices=state["feature_indices"],
        return_full_for_render=state["return_full_for_render"],
    )
    state["pc_tensor"][slot].copy_(torch.from_numpy(pc_plane).to(dtype=state["pc_tensor"].dtype))
    state["labels_memmap"][slot] = int(label)
    if state["return_full_for_render"]:
        state["pc_full_tensor"][slot].copy_(
            torch.from_numpy(pc_full_plane).to(dtype=state["pc_full_tensor"].dtype)
        )
    return slot


def plane_point_cloud_to_grid(point_cloud: torch.Tensor) -> torch.Tensor:
    """Normalize plane-based tensors to (D, H, W)."""
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if point_cloud.ndim == 3:
        return point_cloud.contiguous()
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected point_cloud with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")
    n, d = point_cloud.shape
    side = int(math.isqrt(n))
    assert side * side == n, f"N={n} is not a perfect square"
    return point_cloud.reshape(side, side, d).permute(2, 0, 1).contiguous()


def _select_point_cloud_features_torch(
    point_cloud: torch.Tensor,
    feature_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if feature_indices is None:
        return point_cloud
    if point_cloud.ndim == 2:
        return point_cloud[:, feature_indices]
    if point_cloud.ndim == 3:
        return point_cloud[feature_indices]
    raise ValueError(f"Expected point_cloud with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")


class Class3DGenDataset(Dataset):
    """Wraps Standard3DGenDataset for class-conditional training.

    Filters out samples with class label -1 (noise), looks up class labels,
    optionally selects feature channels, and reshapes plane-ordered rows
    into `(C, H, W)` tensors for training.
    """

    def __init__(
        self,
        base_dataset: Standard3DGenDataset,
        class_map: dict,
        feature_indices: Optional[torch.Tensor] = None,
        return_full_for_render: bool = False,
        preload_to_cpu: bool = False,
        lazy_cache_to_cpu: bool = False,
        cache_dtype: torch.dtype = torch.float32,
        preload_max_samples: int = 0,
        preload_workers: int = 0,
    ):
        """
        Args:
            base_dataset: Standard3DGenDataset instance.
            class_map: Dict mapping "dir/file" keys to class label ints.
            feature_indices: Optional tensor of feature column indices to select
                (e.g. for sh_degree0_only mode).
            return_full_for_render: If True and feature_indices is set, also return
                the full 59-channel plane grid for render loss GT.
            preload_to_cpu: If True, eagerly materialize the transformed training
                samples in CPU memory during initialization.
            lazy_cache_to_cpu: If True, create a shared CPU cache that fills on
                first access instead of preloading everything up front.
            cache_dtype: Tensor dtype used for the shared CPU cache payload.
            preload_max_samples: Limit eager CPU preload to the first N samples.
                `0` means preload the full dataset.
            preload_workers: Number of worker processes to use while building the
                shared preload cache. `0` selects an automatic value.
        """
        if preload_to_cpu and lazy_cache_to_cpu:
            raise ValueError("preload_to_cpu and lazy_cache_to_cpu are mutually exclusive")
        if preload_max_samples < 0:
            raise ValueError("preload_max_samples must be >= 0")
        base_order = getattr(base_dataset, "point_cloud_order", "plane")
        if base_order != "plane":
            raise ValueError(
                "Class3DGenDataset expects a plane-ordered base dataset; "
                f"got point_cloud_order={base_order!r}"
            )
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.feature_indices = feature_indices
        self.return_full_for_render = return_full_for_render and (feature_indices is not None)
        self.preload_to_cpu = preload_to_cpu
        self.lazy_cache_to_cpu = lazy_cache_to_cpu
        self.cache_dtype = cache_dtype
        self.preload_max_samples = preload_max_samples
        self.preload_workers = preload_workers
        self.cached_pc = None
        self.cached_pc_full = None
        self.cached_labels = None
        self.cached_hash_keys = None
        self.cached_ready = None
        self.cached_sample_count = 0
        self.cache_dir = None
        self.cache_mode = None
        self._cache_backing = {}
        self._feature_indices_np = (
            self.feature_indices.cpu().numpy() if self.feature_indices is not None else None
        )

        # Build index of valid samples (class label != -1)
        self.valid_indices = []
        self.valid_labels = []
        skipped = 0
        for idx in range(len(base_dataset)):
            hash_key = base_dataset.keys[idx]
            tar_gz_path = base_dataset.obj_data[hash_key]
            class_key = tar_gz_path.replace('.tar.gz', '')
            label = class_map.get(class_key, -1)
            if label != -1:
                self.valid_indices.append(idx)
                self.valid_labels.append(label)
            else:
                skipped += 1

        logger.info(
            f"Class3DGenDataset: {len(self.valid_indices)} valid samples, "
            f"{skipped} skipped (class -1 or missing)"
        )

        if self.preload_to_cpu:
            self._attach_or_build_preload_cache()
        elif self.lazy_cache_to_cpu:
            self._attach_or_build_lazy_cache()

    def __len__(self):
        if self.preload_to_cpu and self.cached_sample_count > 0:
            return self.cached_sample_count
        return len(self.valid_indices)

    def _build_sample(self, real_idx, label: Optional[int] = None):
        sample = self.base_dataset[real_idx]
        hash_key = sample['hash_key']

        # Get class label
        if label is None:
            tar_gz_path = self.base_dataset.obj_data[self.base_dataset.keys[real_idx]]
            class_key = tar_gz_path.replace('.tar.gz', '')
            label = self.class_map[class_key]

        # Point cloud from Standard3DGenDataset is already plane-based: (C, H, W)
        # for current data, but keep 2D flat-row compatibility for older outputs.
        pc_full = sample['point_cloud']

        # Select features if requested
        pc = _select_point_cloud_features_torch(pc_full, self.feature_indices)
        pc = plane_point_cloud_to_grid(pc)
        if self.return_full_for_render:
            pc_full_grid = plane_point_cloud_to_grid(pc_full)

        if self.return_full_for_render:
            return pc, label, pc_full_grid, hash_key

        return pc, label, hash_key

    def _build_cache_key(self, cache_mode: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"class3dgen_preload_v{PRELOAD_CACHE_VERSION}".encode("utf-8"))
        hasher.update(f"cache_mode:{cache_mode}".encode("utf-8"))
        _hash_strings(hasher, self.base_dataset.keys)
        _hash_strings(hasher, [self.base_dataset.obj_data[key] for key in self.base_dataset.keys])
        _hash_array(hasher, np.asarray(self.valid_indices, dtype=np.int64))
        _hash_array(hasher, np.asarray(self.valid_labels, dtype=np.int64))
        if self.feature_indices is None:
            hasher.update(b"feature_indices:none")
        else:
            _hash_array(hasher, self.feature_indices.cpu().numpy())
        if self.base_dataset.mean is None:
            hasher.update(b"mean:none")
        else:
            _hash_array(hasher, self.base_dataset.mean)
        if self.base_dataset.std is None:
            hasher.update(b"std:none")
        else:
            _hash_array(hasher, self.base_dataset.std)
        hasher.update(str(self.base_dataset.gs_path).encode("utf-8"))
        hasher.update(
            str(self.base_dataset.rendering_path).encode("utf-8")
            if self.base_dataset.rendering_path is not None else b"rendering_path:none"
        )
        hasher.update(str(self.base_dataset.num_images).encode("utf-8"))
        hasher.update(b"return_full:1" if self.return_full_for_render else b"return_full:0")
        hasher.update(f"cache_dtype:{_torch_dtype_to_name(self.cache_dtype)}".encode("utf-8"))
        hasher.update(f"preload_max_samples:{self.preload_max_samples}".encode("utf-8"))
        return hasher.hexdigest()[:32]

    def _cache_paths(self, cache_mode: str):
        cache_root = _default_preload_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = self._build_cache_key(cache_mode)
        cache_dir = cache_root / cache_key
        return cache_root, cache_dir, cache_root / f"{cache_key}.lock"

    def _resolved_preload_workers(self, dataset_len: int) -> int:
        if dataset_len <= 1:
            return 1
        if self.preload_workers > 0:
            return min(self.preload_workers, dataset_len)
        cpu_count = os.cpu_count() or 1
        return min(dataset_len, max(1, cpu_count))

    def _resolved_preload_sample_count(self, dataset_len: int) -> int:
        if self.preload_max_samples > 0:
            return min(dataset_len, self.preload_max_samples)
        return dataset_len

    def _attach_shared_cache(self, meta_path: Path, expected_mode: str) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != PRELOAD_CACHE_VERSION:
            raise ValueError(
                f"Unsupported preload cache version {meta.get('version')} at {meta_path}"
            )
        if meta.get("cache_mode") != expected_mode:
            raise ValueError(
                f"Expected cache_mode={expected_mode}, got {meta.get('cache_mode')} at {meta_path}"
            )

        pc_memmap, pc_tensor = _open_tensor_memmap(
            meta["pc_path"],
            mode="r+",
            tensor_dtype=_dtype_name_to_torch(meta["pc_dtype"]),
            shape=tuple(meta["pc_shape"]),
        )
        labels_memmap = np.memmap(
            meta["labels_path"],
            mode="r+",
            dtype=np.dtype(meta["labels_dtype"]),
            shape=tuple(meta["labels_shape"]),
        )
        ready_memmap = np.memmap(
            meta["ready_path"],
            mode="r+",
            dtype=np.dtype(meta["ready_dtype"]),
            shape=tuple(meta["ready_shape"]),
        )
        with open(meta["hash_keys_path"], "r", encoding="utf-8") as f:
            hash_keys = json.load(f)

        self.cached_pc = pc_tensor
        self.cached_labels = torch.from_numpy(labels_memmap)
        self.cached_hash_keys = hash_keys
        self.cached_ready = ready_memmap
        self.cached_sample_count = int(meta.get("cached_sample_count", len(hash_keys)))
        self.cache_dir = meta_path.parent
        self.cache_mode = expected_mode
        self._cache_backing = {
            "pc": pc_memmap,
            "labels": labels_memmap,
            "ready": ready_memmap,
        }

        if self.return_full_for_render:
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                meta["pc_full_path"],
                mode="r+",
                tensor_dtype=_dtype_name_to_torch(meta["pc_full_dtype"]),
                shape=tuple(meta["pc_full_shape"]),
            )
            self.cached_pc_full = pc_full_tensor
            self._cache_backing["pc_full"] = pc_full_memmap
        else:
            self.cached_pc_full = None

    def _build_shared_cache(self, cache_dir: Path, meta_path: Path) -> None:
        preload_start = time.time()
        dataset_len = len(self.valid_indices)
        preload_len = self._resolved_preload_sample_count(dataset_len)
        worker_count = self._resolved_preload_workers(preload_len)
        logger.info(
            "Preloading %d/%d class-conditioned samples into shared CPU cache at %s using %d worker(s)",
            preload_len,
            dataset_len,
            cache_dir,
            worker_count,
        )

        if preload_len == 0:
            raise ValueError("preload_to_cpu is enabled but resolved preload sample count is 0")

        first_real_idx = self.valid_indices[0]
        first_label = self.valid_labels[0]
        first_pc, first_pc_full = _build_preload_grids(
            base_dataset=self.base_dataset,
            real_idx=first_real_idx,
            feature_indices=self._feature_indices_np,
            return_full_for_render=self.return_full_for_render,
        )

        pc_shape = (preload_len,) + tuple(first_pc.shape)
        cache_dtype_name = _torch_dtype_to_name(self.cache_dtype)
        pc_path = cache_dir / "pc.dat"
        labels_path = cache_dir / "labels.dat"
        ready_path = cache_dir / "ready.dat"
        hash_keys_path = cache_dir / "hash_keys.json"

        pc_memmap, pc_tensor = _open_tensor_memmap(
            str(pc_path), mode="w+", tensor_dtype=self.cache_dtype, shape=pc_shape
        )
        labels_memmap = np.memmap(labels_path, mode="w+", dtype=np.int64, shape=(preload_len,))
        ready_memmap = np.memmap(ready_path, mode="w+", dtype=np.uint8, shape=(preload_len,))

        pc_full_memmap = None
        pc_full_tensor = None
        pc_full_path = None
        pc_full_shape = None
        pc_full_dtype_name = None
        if self.return_full_for_render:
            pc_full_shape = (preload_len,) + tuple(first_pc_full.shape)
            pc_full_dtype_name = cache_dtype_name
            pc_full_path = cache_dir / "pc_full.dat"
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                str(pc_full_path),
                mode="w+",
                tensor_dtype=self.cache_dtype,
                shape=pc_full_shape,
            )
        hash_keys = [self.base_dataset.keys[real_idx] for real_idx in self.valid_indices[:preload_len]]
        labels_memmap[:] = np.asarray(self.valid_labels[:preload_len], dtype=np.int64)
        ready_memmap[:] = 0

        def write_sample(slot: int, pc_plane: np.ndarray, label: int, pc_full_plane: Optional[np.ndarray]) -> None:
            pc_tensor[slot].copy_(torch.from_numpy(pc_plane).to(dtype=pc_tensor.dtype))
            labels_memmap[slot] = int(label)
            if self.return_full_for_render:
                pc_full_tensor[slot].copy_(torch.from_numpy(pc_full_plane).to(dtype=pc_full_tensor.dtype))

        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=dataset_len,
                desc="preload_to_cpu",
                unit="sample",
                dynamic_ncols=True,
            )

        try:
            write_sample(0, first_pc, first_label, first_pc_full)
            if progress is not None:
                progress.update(1)
            task_iter = (
                (
                    slot,
                    real_idx,
                    label,
                )
                for slot, (real_idx, label) in enumerate(
                    zip(self.valid_indices[1:preload_len], self.valid_labels[1:preload_len]), start=1
                )
            )
            if worker_count == 1:
                for task in task_iter:
                    _preload_worker_write_sample_local = _build_preload_grids(
                        base_dataset=self.base_dataset,
                        real_idx=task[1],
                        feature_indices=self._feature_indices_np,
                        return_full_for_render=self.return_full_for_render,
                    )
                    write_sample(task[0], _preload_worker_write_sample_local[0], task[2], _preload_worker_write_sample_local[1])
                    if progress is not None:
                        progress.update(1)
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_preload_worker,
                    initargs=(
                        self.base_dataset,
                        self._feature_indices_np,
                        self.return_full_for_render,
                        str(pc_path),
                        pc_shape,
                        cache_dtype_name,
                        str(labels_path),
                        (preload_len,),
                        "int64",
                        str(pc_full_path) if pc_full_path is not None else None,
                        pc_full_shape,
                        pc_full_dtype_name,
                    ),
                ) as executor:
                    pending = set()
                    max_pending = max(1, worker_count * 2)

                    def submit_next_task() -> bool:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            return False
                        pending.add(executor.submit(_preload_worker_write_sample, task))
                        return True

                    for _ in range(max_pending):
                        if not submit_next_task():
                            break

                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                            if progress is not None:
                                progress.update(1)
                            submit_next_task()
        finally:
            if progress is not None:
                progress.close()

        ready_memmap[:] = 1
        pc_memmap.flush()
        labels_memmap.flush()
        ready_memmap.flush()
        if pc_full_memmap is not None:
            pc_full_memmap.flush()

        hash_keys_tmp = hash_keys_path.with_suffix(".json.tmp")
        with open(hash_keys_tmp, "w", encoding="utf-8") as f:
            json.dump(hash_keys, f)
        os.replace(hash_keys_tmp, hash_keys_path)

        meta = {
            "version": PRELOAD_CACHE_VERSION,
            "cache_mode": "preload",
            "pc_path": str(pc_path),
            "pc_dtype": cache_dtype_name,
            "pc_shape": list(pc_shape),
            "labels_path": str(labels_path),
            "labels_dtype": "int64",
            "labels_shape": [preload_len],
            "ready_path": str(ready_path),
            "ready_dtype": "uint8",
            "ready_shape": [preload_len],
            "cached_sample_count": preload_len,
            "hash_keys_path": str(hash_keys_path),
            "return_full_for_render": self.return_full_for_render,
        }
        if self.return_full_for_render:
            meta.update(
                {
                    "pc_full_path": str(pc_full_path),
                    "pc_full_dtype": pc_full_dtype_name,
                    "pc_full_shape": list(pc_full_shape),
                }
            )

        meta_tmp = meta_path.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)

        total_bytes = int(np.prod(pc_shape)) * _element_size_bytes(self.cache_dtype)
        if self.return_full_for_render:
            total_bytes += int(np.prod(pc_full_shape)) * _element_size_bytes(self.cache_dtype)
        elapsed = time.time() - preload_start
        logger.info(
            "Finished shared CPU preload: %d/%d samples cached in %.1fs (dtype=%s, tensor storage %.2f GiB)",
            preload_len,
            dataset_len,
            elapsed,
            cache_dtype_name,
            total_bytes / (1024 ** 3),
        )

    def _build_lazy_cache(self, cache_dir: Path, meta_path: Path) -> None:
        lazy_start = time.time()
        logger.info(
            "Initializing lazy class-conditioned CPU cache at %s",
            cache_dir,
        )

        first_real_idx = self.valid_indices[0]
        first_label = self.valid_labels[0]
        first_pc, first_pc_full = _build_preload_grids(
            base_dataset=self.base_dataset,
            real_idx=first_real_idx,
            feature_indices=self._feature_indices_np,
            return_full_for_render=self.return_full_for_render,
        )
        dataset_len = len(self.valid_indices)

        pc_shape = (dataset_len,) + tuple(first_pc.shape)
        cache_dtype_name = _torch_dtype_to_name(self.cache_dtype)
        pc_path = cache_dir / "pc.dat"
        labels_path = cache_dir / "labels.dat"
        ready_path = cache_dir / "ready.dat"
        hash_keys_path = cache_dir / "hash_keys.json"

        pc_memmap, pc_tensor = _open_tensor_memmap(
            str(pc_path), mode="w+", tensor_dtype=self.cache_dtype, shape=pc_shape
        )
        labels_memmap = np.memmap(labels_path, mode="w+", dtype=np.int64, shape=(dataset_len,))
        ready_memmap = np.memmap(ready_path, mode="w+", dtype=np.uint8, shape=(dataset_len,))

        pc_full_memmap = None
        pc_full_tensor = None
        pc_full_path = None
        pc_full_shape = None
        pc_full_dtype_name = None
        if self.return_full_for_render:
            pc_full_shape = (dataset_len,) + tuple(first_pc_full.shape)
            pc_full_dtype_name = cache_dtype_name
            pc_full_path = cache_dir / "pc_full.dat"
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                str(pc_full_path),
                mode="w+",
                tensor_dtype=self.cache_dtype,
                shape=pc_full_shape,
            )

        labels_memmap[:] = np.asarray(self.valid_labels, dtype=np.int64)
        ready_memmap[:] = 0
        pc_tensor[0].copy_(torch.from_numpy(first_pc).to(dtype=pc_tensor.dtype))
        if self.return_full_for_render:
            pc_full_tensor[0].copy_(torch.from_numpy(first_pc_full).to(dtype=pc_full_tensor.dtype))
        ready_memmap[0] = 1

        pc_memmap.flush()
        labels_memmap.flush()
        ready_memmap.flush()
        if pc_full_memmap is not None:
            pc_full_memmap.flush()

        hash_keys = [self.base_dataset.keys[real_idx] for real_idx in self.valid_indices]
        hash_keys_tmp = hash_keys_path.with_suffix(".json.tmp")
        with open(hash_keys_tmp, "w", encoding="utf-8") as f:
            json.dump(hash_keys, f)
        os.replace(hash_keys_tmp, hash_keys_path)

        meta = {
            "version": PRELOAD_CACHE_VERSION,
            "cache_mode": "lazy",
            "pc_path": str(pc_path),
            "pc_dtype": cache_dtype_name,
            "pc_shape": list(pc_shape),
            "labels_path": str(labels_path),
            "labels_dtype": "int64",
            "labels_shape": [dataset_len],
            "ready_path": str(ready_path),
            "ready_dtype": "uint8",
            "ready_shape": [dataset_len],
            "cached_sample_count": dataset_len,
            "hash_keys_path": str(hash_keys_path),
            "return_full_for_render": self.return_full_for_render,
        }
        if self.return_full_for_render:
            meta.update(
                {
                    "pc_full_path": str(pc_full_path),
                    "pc_full_dtype": pc_full_dtype_name,
                    "pc_full_shape": list(pc_full_shape),
                }
            )

        meta_tmp = meta_path.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)

        total_bytes = int(np.prod(pc_shape)) * _element_size_bytes(self.cache_dtype)
        if self.return_full_for_render:
            total_bytes += int(np.prod(pc_full_shape)) * _element_size_bytes(self.cache_dtype)
        elapsed = time.time() - lazy_start
        logger.info(
            "Initialized lazy shared CPU cache in %.1fs (dtype=%s, reserved tensor storage %.2f GiB)",
            elapsed,
            cache_dtype_name,
            total_bytes / (1024 ** 3),
        )

    def _attach_or_build_preload_cache(self) -> None:
        cache_root, cache_dir, lock_path = self._cache_paths("preload")
        meta_path = cache_dir / "meta.json"
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if meta_path.exists():
                try:
                    self._attach_shared_cache(meta_path, expected_mode="preload")
                    logger.info("Attached to existing shared CPU preload cache: %s", cache_dir)
                    return
                except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Shared preload cache at %s is invalid (%s); rebuilding",
                        cache_dir,
                        exc,
                    )
                    meta_path.unlink(missing_ok=True)

            cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_shared_cache(cache_dir, meta_path)
            self._attach_shared_cache(meta_path, expected_mode="preload")

    def _attach_or_build_lazy_cache(self) -> None:
        cache_root, cache_dir, lock_path = self._cache_paths("lazy")
        meta_path = cache_dir / "meta.json"
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if meta_path.exists():
                try:
                    self._attach_shared_cache(meta_path, expected_mode="lazy")
                    logger.info("Attached to existing lazy shared CPU cache: %s", cache_dir)
                    return
                except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Lazy shared CPU cache at %s is invalid (%s); rebuilding",
                        cache_dir,
                        exc,
                    )
                    meta_path.unlink(missing_ok=True)

            cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_lazy_cache(cache_dir, meta_path)
            self._attach_shared_cache(meta_path, expected_mode="lazy")

    def _lazy_cache_lock_path(self, idx: int) -> Path:
        stripe = idx % LAZY_CACHE_LOCK_STRIPES
        return self.cache_dir / f"lazy_cache_{stripe:03d}.lock"

    def _ensure_lazy_sample_cached(self, idx: int) -> None:
        if not self.lazy_cache_to_cpu or self.cached_pc is None:
            return
        if int(self.cached_ready[idx]) == 1:
            return

        lock_path = self._lazy_cache_lock_path(idx)
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if int(self.cached_ready[idx]) == 1:
                return

            real_idx = self.valid_indices[idx]
            pc_plane, pc_full_plane = _build_preload_grids(
                base_dataset=self.base_dataset,
                real_idx=real_idx,
                feature_indices=self._feature_indices_np,
                return_full_for_render=self.return_full_for_render,
            )
            self.cached_pc[idx].copy_(torch.from_numpy(pc_plane).to(dtype=self.cached_pc.dtype))
            if self.return_full_for_render:
                self.cached_pc_full[idx].copy_(
                    torch.from_numpy(pc_full_plane).to(dtype=self.cached_pc_full.dtype)
                )
            self.cached_ready[idx] = 1

    def __getitem__(self, idx):
        if self.cached_pc is not None:
            if idx >= self.cached_sample_count:
                if self.preload_to_cpu:
                    raise IndexError(
                        f"idx={idx} is outside the eagerly preloaded range "
                        f"(cached_sample_count={self.cached_sample_count})"
                    )
                real_idx = self.valid_indices[idx]
                return self._build_sample(real_idx, label=self.valid_labels[idx])
            if self.lazy_cache_to_cpu and int(self.cached_ready[idx]) == 0:
                self._ensure_lazy_sample_cached(idx)
            label = int(self.cached_labels[idx])
            hash_key = self.cached_hash_keys[idx]
            if self.return_full_for_render:
                return self.cached_pc[idx], label, self.cached_pc_full[idx], hash_key
            return self.cached_pc[idx], label, hash_key

        real_idx = self.valid_indices[idx]
        return self._build_sample(real_idx, label=self.valid_labels[idx])
