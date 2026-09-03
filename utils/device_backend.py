"""Small backend abstraction so DARTree can run on NVIDIA CUDA or Huawei Ascend NPU.

The DARTree algorithm is written with ``torch.cuda`` / CUDA-graph / NVIDIA-Triton
fast paths.  To keep the rest of the source unchanged, all *strongly* NVIDIA-bound
operations are routed through this module:

* Accelerator selection (``set_device``), stream synchronisation, seeding and
  wall-clock timing helpers.
* A query for whether CUDA-graph / NVIDIA-Triton fast paths may be used.  They are
  only enabled on real CUDA devices.  On Ascend NPUs the already-existing pure
  PyTorch eager fallbacks in the code are used instead.

``torch_npu`` is imported lazily/optionally so that importing this tree never fails
on a machine that has no Ascend toolkit installed (e.g. an NVIDIA-only box).  On a
CANN/torch_npu Linux host it registers the ``npu`` device type exactly like
``import torch_npu``.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import torch

DeviceLike = Union[torch.device, str, int, None]

try:  # Optional: present only on CANN/Ascend hosts.
    import torch_npu as _torch_npu  # type: ignore
except Exception:  # pragma: no cover - exercised only when torch_npu is missing
    _torch_npu = None


def torch_npu_available() -> bool:
    """Return True only when the Ascend NPU backend is importable and usable."""
    if _torch_npu is None:
        return False
    try:
        return bool(_torch_npu.npu.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def device_type(device: DeviceLike) -> str:
    """Return the accelerator type ('cuda' | 'npu' | 'cpu') without raising on
    an unregistered backend (torch.device('npu:0') needs torch_npu imported)."""
    if isinstance(device, torch.device):
        return device.type
    text = str(device)
    if text.startswith("npu"):
        return "npu"
    if text.startswith("cuda"):
        return "cuda"
    try:
        return torch.device(text).type
    except Exception:
        return "cpu"


def is_cuda_device(device: DeviceLike) -> bool:
    return device_type(device) == "cuda"


def is_npu_device(device: DeviceLike) -> bool:
    return device_type(device) == "npu"


def is_accelerator_device(device: DeviceLike) -> bool:
    return device_type(device) in ("cuda", "npu")


def accelerator_available(device: DeviceLike) -> bool:
    kind = device_type(device)
    if kind == "cuda":
        return cuda_available()
    if kind == "npu":
        return torch_npu_available()
    return False


def set_device(device: DeviceLike) -> None:
    """Select the default accelerator device for the active backend."""
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.set_device(device)
    elif kind == "npu":
        if _torch_npu is None:
            raise RuntimeError(
                "torch_npu is required to select an Ascend NPU device; "
                "install it with your CANN toolkit before using --device npu:0."
            )
        _torch_npu.npu.set_device(device)
    else:
        raise ValueError(
            f"Unsupported device {device!r}: expected cuda:<i> or npu:<i>."
        )


def synchronize(device: DeviceLike = None) -> None:
    """Block until kernels on ``device`` (or the default accelerator) complete."""
    if device is None:
        if cuda_available():
            torch.cuda.synchronize()
        elif torch_npu_available():
            _torch_npu.npu.synchronize()
        return
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "npu":
        if _torch_npu is not None:
            _torch_npu.npu.synchronize(device)


def wall_time(device: DeviceLike = None) -> float:
    """Synchronise the accelerator (if any) then return wall-clock seconds.

    Mirrors the old ``cuda_time`` helper used throughout DARTree, generalised to
    also synchronise Ascend NPUs so timing stats stay meaningful there.
    """
    synchronize(device)
    return time.perf_counter()


def seed_all(seed: int) -> None:
    """Seed torch + the active accelerator(s) deterministically."""
    torch.manual_seed(int(seed))
    if cuda_available():
        torch.cuda.manual_seed_all(int(seed))
    if _torch_npu is not None and torch_npu_available():
        try:
            _torch_npu.npu.manual_seed_all(int(seed))
        except Exception:
            # Some torch_npu builds do not expose manual_seed_all; seeding the
            # NPU generator is then left to torch_npu's own defaults.
            pass


# ---------------------------------------------------------------------------
# NVIDIA-only fast-path queries (CUDA graphs + NVIDIA-oriented Triton kernels).
# ---------------------------------------------------------------------------

def can_use_cuda_graphs(device: DeviceLike) -> bool:
    """CUDA-graph capture is only supported on a real CUDA device."""
    return is_cuda_device(device) and cuda_available()


def can_use_nvidia_triton() -> bool:
    """The inline Triton kernels in this repo are tuned for NVIDIA backends;
    never run them on an Ascend NPU (torch_npu may even ship its own Triton)."""
    return cuda_available()
