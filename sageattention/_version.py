"""
Build fingerprint and runtime guard for the AMD/gfx12 SageAttention port.

Why this file exists
--------------------
The gfx12 native extension (`_qattn_gfx12_native*.pyd`) statically embeds device
code and links the PyTorch C++ ABI (`torch_cpu.dll`, `c10.dll`, `torch_python.dll`,
`amdhip64_*.dll`).  Those symbols are version-coupled: a wheel built against a
different torch fails to load with

    ImportError: DLL load failed while importing _qattn_gfx12_native:
    The specified procedure could not be found.

`core.py` catches that failure to stay importable and sets
`GFX12_NATIVE_ENABLED = False`.  In ComfyUI that is a silent downgrade:
`attention_sage` sees an exception on the first real call, logs "Error running
sage attention: ..." and quietly falls back to PyTorch attention.  The user
pays for a GPU kernel and gets none of it, with no clear diagnostic.

This module converts that silent failure into a hard, actionable ImportError at
`import sageattention` time on a gfx12 machine.

The `BUILD_*` values below are the toolchain the shipped wheel was built with.
`BUILD_TORCH` is compared against the installed torch at import time; the rest
are informational.  Update them when rebuilding against a different torch.
"""

from __future__ import annotations

import os

# --- Build fingerprint ------------------------------------------------------
BUILD_TORCH = "2.13.0+rocm10.0.0"
BUILD_ROCM = "7.15.26333"
BUILD_ARCH = "gfx1201"
BUILD_PYTHON = "3.12.10"
BUILD_GLIBCXX_USE_CXX11_ABI = True
BUILD_MSVC = "14.50.35717"
BUILD_ID = "sageattention-gfx12/2.2.0+amd.gfx12.1"

# The two extension modules that must be present for a complete install.
_REQUIRED_EXTENSIONS = ("_qattn_gfx12_native", "_fused")


class SageAttentionABIError(ImportError):
    """The native extension is present but cannot be loaded (torch ABI skew)."""


class SageAttentionBuildError(ImportError):
    """The native extension was never built into this installation."""


def extension_path(module_name: str) -> str:
    """Absolute path of a packaged extension module, whether or not it exists."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), module_name)


def extension_present(module_name: str) -> bool:
    """True when a compiled extension file for this interpreter exists on disk.

    Checked by filename rather than by import, so that "the file is missing"
    and "the file is present but will not load" produce different errors.
    """
    import importlib.machinery

    base = extension_path(module_name)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if os.path.exists(base + suffix):
            return True
    return False


def device_archs() -> list:
    """gcnArchName of every visible HIP device.

    Deliberately uses `torch.cuda.get_device_properties`, which reads the device
    table and does NOT create a context or allocate. `torch.cuda.is_available()`
    is avoided on purpose: with a wedged HIP runtime, any call that touches the
    memory allocator can abort the process (c10 AbortHandler -> RaiseException),
    and `is_available()` can probe more than it needs to. The runtime guard must
    never be the thing that kills an otherwise fine import.

    Returns an empty list when no HIP runtime/GPU is visible, which is not an
    error: a CPU-only import must still succeed.
    """
    try:
        import torch

        if torch.version.hip is None:
            return []
        archs = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            arch = getattr(props, "gcnArchName", "") or ""
            archs.append(arch.split(":", 1)[0])
        return archs
    except Exception:
        return []


def check_runtime(raise_on_failure: bool = True) -> dict:
    """Validate the install against the triple it was built for.

    Raises SageAttentionABIError / SageAttentionBuildError on a gfx12 machine
    when the native extension is unusable.  On a non-gfx12 machine the missing
    extension is reported in the returned dict but does not raise, because
    nothing in this wheel is expected to work there anyway.

    Set SAGEATTENTION_ALLOW_UNVERIFIED_TORCH=1 to downgrade a torch-version
    mismatch to a warning (for users who knowingly run an untested torch).
    """
    import torch

    status = {
        "build_id": BUILD_ID,
        "torch": torch.__version__,
        "build_torch": BUILD_TORCH,
        "hip": torch.version.hip,
        "build_rocm": BUILD_ROCM,
        "device_archs": device_archs(),
        "extension_present": {m: extension_present(m) for m in _REQUIRED_EXTENSIONS},
    }
    status["gfx12_device"] = any(a.startswith("gfx12") for a in status["device_archs"])

    from . import core

    status["gfx12_native_enabled"] = bool(core.GFX12_NATIVE_ENABLED)
    status["gfx12_import_error"] = core.GFX12_NATIVE_IMPORT_ERROR

    if not raise_on_failure:
        return status

    allow_unverified = os.environ.get(
        "SAGEATTENTION_ALLOW_UNVERIFIED_TORCH", ""
    ).strip().lower() in {"1", "true", "yes"}

    if torch.__version__ != BUILD_TORCH and not allow_unverified:
        raise SageAttentionABIError(
            f"{BUILD_ID} was built against torch=={BUILD_TORCH} but this "
            f"interpreter has torch=={torch.__version__}.\n"
            f"The native extension statically links the PyTorch C++ ABI, so the "
            f"usual symptom is 'ImportError: DLL load failed ... The specified "
            f"procedure could not be found', after which GFX12_NATIVE_ENABLED "
            f"silently becomes False and ComfyUI falls back to PyTorch attention.\n"
            f"Fix: rebuild this wheel against the installed torch, or install "
            f"torch=={BUILD_TORCH}. Set SAGEATTENTION_ALLOW_UNVERIFIED_TORCH=1 "
            f"to bypass this check (unsupported)."
        )

    if status["gfx12_native_enabled"]:
        return status

    if not status["gfx12_device"]:
        # Not a gfx12 machine; the missing kernel is expected and harmless here.
        return status

    missing = [m for m, present in status["extension_present"].items() if not present]
    import_error = status["gfx12_import_error"]

    if missing:
        raise SageAttentionBuildError(
            f"{BUILD_ID} is installed on a gfx12 device "
            f"({', '.join(status['device_archs'])}) but the compiled extension(s) "
            f"{missing} are not present in "
            f"{os.path.dirname(os.path.abspath(__file__))}.\n"
            f"This installation was built without a gfx12 target. Rebuild with "
            f"PYTORCH_ROCM_ARCH={BUILD_ARCH} against torch=={BUILD_TORCH}, or "
            f"install a wheel that contains both .pyd files."
        )

    raise SageAttentionABIError(
        f"{BUILD_ID} is installed on a gfx12 device "
        f"({', '.join(status['device_archs'])}) and both extension files exist, "
        f"but importing them failed:\n"
        f"    {type(import_error).__name__}: {import_error}\n"
        f"Refusing to continue: the historical behaviour was to set "
        f"GFX12_NATIVE_ENABLED=False and let ComfyUI silently fall back to "
        f"PyTorch attention, which hides the failure.\n"
        f"Most likely cause: torch C++ ABI skew (wheel built for "
        f"torch=={BUILD_TORCH}, interpreter has torch=={torch.__version__}). "
        f"Check the missing symbols with pefile, then rebuild against the "
        f"installed torch."
    )
