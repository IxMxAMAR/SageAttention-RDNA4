"""
SageAttention for AMD gfx12 (RDNA4): native port.

Drop-in replacement for the upstream `sageattention` distribution. The only
name ComfyUI imports is `sageattn`, and the only capability it probes is
whether `sageattn` advertises an `attn_mask` parameter. `sageattn` here
deliberately does not, so ComfyUI routes masked calls to `attention_pytorch`
instead of forwarding a mask this kernel would silently ignore.

Public surface is unchanged from upstream/PR #368: the same six names.
"""

from ._version import (  # noqa: F401
    BUILD_ARCH,
    BUILD_ID,
    BUILD_ROCM,
    BUILD_TORCH,
    SageAttentionABIError,
    SageAttentionBuildError,
    check_runtime,
)

# Validate the build triple and the native extension *before* pulling in the
# dispatcher, so a torch ABI mismatch is a loud, single-cause ImportError rather
# than a silent `GFX12_NATIVE_ENABLED = False` followed by a fallback.
check_runtime()

from .core import sageattn, sageattn_varlen  # noqa: E402
from .core import sageattn_qk_int8_pv_fp16_triton  # noqa: E402
from .core import sageattn_qk_int8_pv_fp16_cuda  # noqa: E402
from .core import sageattn_qk_int8_pv_fp8_cuda  # noqa: E402
from .core import sageattn_qk_int8_pv_fp8_cuda_sm90  # noqa: E402
from .core import sageattn_qk_int8_pv_gfx12_native  # noqa: E402
from .core import GFX12_NATIVE_ENABLED  # noqa: E402

__version__ = "2.2.0+amd.gfx12.2"

__all__ = [
    "sageattn",
    "sageattn_varlen",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
    "sageattn_qk_int8_pv_gfx12_native",
    "GFX12_NATIVE_ENABLED",
    "__version__",
]
