"""Prebuilt SK1 backend for the gfx12 native path (on by default on gfx1201).

This package ships a prebuilt gfx1201 code object plus the loader that launches it through the HIP
runtime DLL that torch has already loaded. Three code objects are packaged:

* `sk1_t4a1s.gfx1201.hsaco` serves contiguous HND calls. It takes a per-(b,h) scale `S` as a 12th
  argument. A key token with `|V| > 448` and enough softmax weight used to overflow the fp8
  conversion in the P-fold, because `p8 = e4m3(p * sv * 448) = e4m3(p * amax_v)` had no clamp, and the
  output went non-finite (a Krea2 render went black from one step on). The kernel now takes
  `S = clamp_min(max_j sv(j), 1)` computed on the device, stages `SV' = SV/S`, and applies `x S` in
  the epilogue. When every `amax_v <= 448`, `S == 1.0` exactly and the result is bit-identical to
  the earlier `sk1_t4a1` kernel.
* `sk1_t4a1n.gfx1201.hsaco` is the same kernel with six extra integer strides for Q and O. It serves
  strided HND and NHD inputs without a copy.
* `sk1_t4a1.gfx1201.hsaco` is the earlier kernel without the overflow fix. It is kept as a reference
  and for A/B timing, and nothing routes to it.

`sk1_t4a1` is `sk1_t1` with the 32-element mask/skip select group placed behind one CTA-uniform
`if (need_mask)` branch. It is bit-identical to `sk1_t1` on the accuracy and edge-length test cells
and a few percent faster.

Why a prebuilt code object and not a compiled extension: a `.hsaco` is a self-contained
`elf64-amdgpu` file with no C++ imports, so the `c10` symbol ABI mismatch that breaks a `.pyd`
extension against a different torch build cannot arise. The loader is a plain `ctypes` call with
integer and pointer arguments.

## Default semantics

The backend is used only when all of these hold: the device arch is `gfx1201`, a HIP runtime
resolves, the code object loads, and the call is inside `ENVELOPE`. Any other case falls back to the
shipped path (the gfx12 native extension on gfx12, Triton on other arches), with at most one short
log line per process, and never raises at call time.

* `SAGEATTN_SK1_BACKEND=1` forces it on; a load failure then raises.
* `SAGEATTN_SK1_BACKEND=0` forces it off (the shipped path).
* Unset means on.

The `.hsaco` files are gfx1201-only. There is no in-wheel rebuild path; `tools/build_hsaco.py` in the
source repository rebuilds them from `kernels/hip/`. The arch gate keeps a non-gfx1201 device on the
shipped path instead of raising.

The HIP runtime is resolved at call time and no install path is hard-coded. `rocm_bin_dir()` tries,
in order: `$SK1_HIP_DLL`, the `amdhip64*.dll` already loaded in this process (torch's own copy),
`$SK1_ROCM_BIN`, then the `_rocm_sdk_core` / `_rocm_sdk_devel` layouts under `site-packages`. The
`_rocm_sdk_core` copy is preferred because it is the one torch loads; `_rocm_sdk_devel` is a different
build, and loading it would put a second HIP runtime in the process (see `sk1_loader.py`).
"""
from __future__ import annotations

import os
import sys

# `quant_kv_v8t` does `from quant_triton import ...` as a flat top-level import, so the package
# directory must be importable as a top-level location before it is imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = ["available", "try_sk1_t1", "SK1_HSACO_NAME", "ENVELOPE", "SUPPORTED_ARCHS",
           "device_arch", "arch_supported"]

#: The earlier, unfixed object. Kept for reference; nothing routes to it.
SK1_HSACO_NAME = "sk1_t4a1.gfx1201.hsaco"
SK1_HSACO_PATH = os.path.join(_HERE, SK1_HSACO_NAME)

#: The packaged object is `sk1_t4a1s` (12 arguments: `S` between `VMEAN` and `O`). The earlier
#: object `sk1_t4a1.gfx1201.hsaco` is kept in the package for reference and for A/B timing; nothing
#: routes to it.
SK1_PACKAGED_WITH_S = True
SK1_HSACO_NAME_S = "sk1_t4a1s.gfx1201.hsaco"
SK1_HSACO_PATH_S = os.path.join(_HERE, SK1_HSACO_NAME_S)
SK1_SYMBOLS_S = ("sk1t4a1s_attn_fwd_c0", "sk1t4a1s_attn_fwd_c1")

#: The widened object, `sk1_t4a1n`: the 12 arguments of `sk1_t4a1s` plus six `int` Q/O strides
#: (`kernels/hip/sk1_t4a1n.hip`, derived from `sk1_t4a1s.hip` by textual substitution). It addresses
#: contiguous HND exactly like `sk1_t4a1s`, but contiguous HND calls are still routed to
#: `sk1_t4a1s` so that path is unchanged. The widened object serves only strided HND and NHD calls.
SK1_HSACO_NAME_N = "sk1_t4a1n.gfx1201.hsaco"
SK1_HSACO_PATH_N = os.path.join(_HERE, SK1_HSACO_NAME_N)
SK1_SYMBOLS_N = ("sk1t4a1n_attn_fwd_c0", "sk1t4a1n_attn_fwd_c1")

#: The only arch the packaged code object can load on. The comparison is a prefix test, so a
#: `gcnArchName` such as `gfx1201:sramecc-:xnack-` also matches.
SUPPORTED_ARCHS = ("gfx1201",)

#: The exact envelope `try_sk1_t1` serves.  Anything else returns `None` (caller falls back).
#: `layout` is HND or NHD, and `contiguous` means "contiguous in its own layout, or a physically-NHD
#: view" (`sk1_loader._strided_triple`). Contiguous HND is served by `sk1_t4a1s`; everything else by
#: `sk1_t4a1n`.
ENVELOPE = dict(
    layout=("HND", "NHD"),
    dtype="fp16",
    head_dim=128,
    contiguous=("HND-contiguous", "physically-NHD view (HND or NHD logical order)"),
    causal=(0, 1),
    smooth_k=(False, True),
    batch_heads="any",
    seq_len="N >= 1",
    return_lse=False,
    attn_mask=False,
    smooth_v=False,
)

_BACKEND = None
_BACKEND_FAILED = None
_BACKEND_N = None
_BACKEND_N_FAILED = None
_LOGGED = False


def device_arch(device=None) -> str:
    """The device's `gcnArchName` (prefix before `:`), or `""` if it cannot be read.  Never raises.

    A named function so a test can monkeypatch the arch query without touching the routing logic.
    """
    try:
        import torch
        if device is None:
            idx = 0
        elif isinstance(device, int):
            idx = device
        else:
            idx = device.index if getattr(device, "index", None) is not None else 0
        props = torch.cuda.get_device_properties(idx)
        arch = str(getattr(props, "gcnArchName", "") or "")
        return arch.split(":", 1)[0]
    except Exception:
        return ""


def arch_supported(device=None) -> bool:
    """True iff the device's arch is one the packaged code object was built for."""
    arch = device_arch(device)
    return bool(arch) and any(arch.startswith(a) for a in SUPPORTED_ARCHS)


def _log_once(reason: str) -> None:
    """At most ONE short line per process, and never an exception out of the logging itself."""
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    try:
        sys.stderr.write("[sageattention] SK1 backend not used (%s); using the shipped path.\n"
                         % reason)
        sys.stderr.flush()
    except Exception:                                          # pragma: no cover
        pass


def packaged_hsaco_path() -> str:
    """The object the packaged path loads: `sk1_t4a1s`, or `sk1_t4a1` when `SK1_PACKAGED_WITH_S` is False."""
    return SK1_HSACO_PATH_S if SK1_PACKAGED_WITH_S else SK1_HSACO_PATH


def fold_v_scale(sv):
    """Stage `SV' = SV / S` on the device and return `S` (B*H,) fp32.

    `S = clamp_min(max_j sv(j), 1)` where `sv(j) = max(amax_v(j), 1)/448`, so `S > 1` exactly when
    some key row has `amax_v > 448`.  The kernel's P fold is `p8 = e4m3(p * sv * 448)`, i.e.
    `e4m3(p * amax_v)` with `p <= 1` and no clamp, so an `amax_v > 448` overflows e4m3 and the output
    goes non-finite; staging `sv/S` makes the fold's argument `amax_v/S <= 448` for every key, and
    the kernel's epilogue carries the compensating `x S`.

    Three device ops on a `(B*H, N_pad)` fp32 tensor: an `amax` reduction, a `clamp_min`, and an
    in-place divide. There is no `.item()`, host sync or CPU round-trip.

    `S == 1.0` exactly whenever every `amax_v <= 448`, and `fl(sv/1.0f) == sv` bit-for-bit, so in
    that case `sv` is unchanged and the kernel is bit-identical to `sk1_t4a1`.
    """
    import torch
    s = sv.amax(dim=1).clamp_min(1.0)
    sv.div_(s.unsqueeze(1))
    return s


def available() -> bool:
    """True iff the prebuilt object exists and a HIP runtime can be resolved.  Never raises."""
    global _BACKEND, _BACKEND_FAILED
    if _BACKEND is not None:
        return True
    if _BACKEND_FAILED is not None:
        return False
    if not os.path.isfile(packaged_hsaco_path()):
        _BACKEND_FAILED = "missing %s" % packaged_hsaco_path()
        return False
    try:
        from .sk1_loader import HipRuntime, Sk1Attn  # noqa: F401
    except Exception as e:                                   # pragma: no cover
        _BACKEND_FAILED = "%s: %s" % (type(e).__name__, e)
        return False
    return True


def _load():
    global _BACKEND, _BACKEND_FAILED
    if _BACKEND is not None:
        return _BACKEND
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        rt = HipRuntime()
        if SK1_PACKAGED_WITH_S:
            _BACKEND = Sk1Attn(SK1_HSACO_PATH_S, rt, symbols=SK1_SYMBOLS_S, with_s=True)
        else:
            _BACKEND = Sk1Attn(SK1_HSACO_PATH, rt)
        return _BACKEND
    except Exception as e:
        _BACKEND_FAILED = "%s: %s" % (type(e).__name__, e)
        raise


def _load_n():
    """The widened object `sk1_t4a1n`, or `None` if it is not installed or will not load.

    Kept separate from `_load()` so that code which swaps `SK1_PACKAGED_WITH_S` and `_BACKEND` to
    select a particular object keeps working. A missing widened object is a clean refusal
    (`reason="no_strided_object"`), never a raise on the default path.
    """
    global _BACKEND_N, _BACKEND_N_FAILED
    if _BACKEND_N is not None:
        return _BACKEND_N
    if _BACKEND_N_FAILED is not None:
        return None
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(SK1_HSACO_PATH_N):
            _BACKEND_N_FAILED = "missing %s" % SK1_HSACO_PATH_N
            return None
        _BACKEND_N = Sk1Attn(SK1_HSACO_PATH_N, HipRuntime(), symbols=SK1_SYMBOLS_N,
                             with_s=True, strided=True)
        return _BACKEND_N
    except Exception as e:                                   # pragma: no cover
        _BACKEND_N_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _envelope_ok(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, smooth_v,
                 attn_mask):
    import torch
    if tensor_layout not in ("HND", "NHD"):
        return "layout"
    if return_lse:
        return "return_lse"
    if attn_mask is not None:
        return "attn_mask"
    if smooth_v:
        return "smooth_v"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "device"
    if q.device != k.device or q.device != v.device:
        return "device"
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        return "dtype"
    if not (q.dim() == 4 and k.dim() == 4 and v.dim() == 4):
        return "rank"
    # Normalise to the (B, H, N, D) reading the kernel uses, then check the stride pattern. Requiring
    # `is_contiguous()` would refuse the two layouts ComfyUI actually passes: an HND view made with
    # `rearrange` and NHD `(B, N, H, D)`. Both are physically NHD, so the accepted set is "contiguous
    # in its own layout, or a physically-NHD view", which is exactly the two patterns of
    # `_strided_triple`. An arbitrary stride set is still refused (it could alias, and the launch is raw).
    if tensor_layout == "NHD":
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qq, kk, vv = q, k, v
    for t in (qq, kk, vv):
        if t.stride(3) != 1:
            return "stride_last"
        if not (t.is_contiguous() or t.transpose(1, 2).is_contiguous()):
            return "strided_pattern"
    q, k, v = qq, kk, vv
    if q.size(3) != 128 or k.size(3) != 128 or v.size(3) != 128:
        return "head_dim"
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        return "batch"
    if q.size(1) != k.size(1) or q.size(1) != v.size(1):
        return "heads"
    # q_len must equal k_len. The kernel takes a single `N` from `q` and pads K/V to `ceil(N/64)*64`,
    # so a longer K would be silently truncated (the fused prologue reads only `n_pad` rows) and a
    # shorter K would read out of bounds.
    if q.size(2) != k.size(2):
        return "seq_len"
    if k.size(2) != v.size(2):
        return "seq_len"
    if q.size(2) < 1 or k.size(2) < 1:
        return "seq_len"
    return None


def try_sk1_t1(q, k, v, *, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False,
               smooth_k=True, smooth_v=False, attn_mask=None, strict=False):
    """Run the prebuilt SK1 kernel if the call is inside `ENVELOPE` and the platform supports it.

    Returns `(out, reason)` where `out` is the attention output in the caller's (B,H,N,D) shape, or
    `(None, reason)` when the call is not served and the caller must fall back to the shipped path.

    `strict=False` (the default): a non-`gfx1201` arch, an unresolvable HIP
    runtime, a code object that will not load, a failure inside the fused prologue, or a launch
    failure **falls back** with at most one short log line per process and **never raises**.
    `strict=True` (`SAGEATTN_SK1_BACKEND=1`, or `sk1_backend=True`): the same failures **raise**,
    because silently falling back on a kernel that was explicitly demanded would hide a defect.
    """
    reason = _envelope_ok(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k,
                          smooth_v, attn_mask)
    if reason is not None:
        return None, reason
    if not arch_supported(q.device):
        if strict:
            raise RuntimeError(
                "SK1 backend forced on (SAGEATTN_SK1_BACKEND=1 / sk1_backend=True) but the device "
                "arch is %r; the packaged code object supports %s"
                % (device_arch(q.device) or "<unknown>", ", ".join(SUPPORTED_ARCHS)))
        _log_once("device arch %r is not %s" % (device_arch(q.device) or "<unknown>",
                                                "/".join(SUPPORTED_ARCHS)))
        return None, "arch"
    # Routing. Contiguous HND (all three tensors) goes to `sk1_t4a1s` on the plain path below.
    # Everything else (strided HND, NHD) goes to the widened `sk1_t4a1n`.
    if tensor_layout == "NHD":
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qq, kk, vv = q, k, v
    contiguous_hnd = bool(tensor_layout == "HND" and q.is_contiguous() and k.is_contiguous()
                          and v.is_contiguous())
    try:
        attn = _load() if contiguous_hnd else _load_n()
    except Exception as e:
        if strict:
            raise
        _log_once("load failed: %s" % e)
        return None, "load_failed"
    if attn is None:
        if strict:
            raise RuntimeError("the widened SK1 object %s is not installed (%s)"
                               % (SK1_HSACO_NAME_N, _BACKEND_N_FAILED))
        _log_once("widened object unavailable: %s" % _BACKEND_N_FAILED)
        return None, "no_strided_object"
    # Everything below can fail for reasons that have nothing to do with the code object: a broken
    # `quant_kv_v8t` module (a partial install), an unsupported platform inside the Triton prologue, or
    # an allocation failure. The default path must therefore stay raise-free all the way to the launch.
    try:
        import torch
        from . import quant_kv_v8t as QV
        B, H, N, D = qq.shape
        sm = float(sm_scale if sm_scale is not None else D ** -0.5)
        n_pad = QV.n_pad_for(N)
        BH = B * H
        if contiguous_hnd:
            qr = q.reshape(BH, N, D).contiguous()
            _, _, k8, sk, v8t, sv = QV.prologue_fp8_fused(q, k, v, n_pad=n_pad,
                                                          smooth_k=bool(smooth_k))
            out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        else:
            # The widened prologue reads K/V straight out of the strided view (no copy) and skips
            # quantising Q, whose int8 result and scales the SK1 kernel would discard anyway.
            from . import quant_kv_v8t_n as QN
            k8, sk, v8t, sv = QN.prologue_fp8_fused_strided(kk, vv, n_pad=n_pad,
                                                            smooth_k=bool(smooth_k))
            if tensor_layout == "NHD":
                out = torch.empty(B, N, H, D, device=q.device, dtype=torch.float16)
            else:
                out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        # Only the 12-argument objects want `S`. When they do, this stages `SV' = SV/S` and returns S
        # for the kernel's epilogue. For every input with |V| <= 448, S == 1.0 and `sv` comes out
        # bit-for-bit unchanged.
        sv_s = fold_v_scale(sv) if attn.with_s else None
    except Exception as e:                                          # noqa: BLE001
        if strict:
            raise
        _log_once("prologue failed: %s" % e)
        return None, "prologue_failed"
    try:
        if contiguous_hnd:
            attn.launch(qr, k8, sk, v8t, sv, None, out, H, N, n_pad, sm, bool(is_causal), s=sv_s)
        else:
            # the widened kernel is addressed in the logical (B, H, N, D) reading of the caller's
            # tensor: for NHD that is the transpose view; `out` is written with the same pattern.
            attn.launch_strided(qq, k8, sk, v8t, sv, None,
                                out.view(B, H, N, D) if tensor_layout == "HND"
                                else out.transpose(1, 2),
                                H, N, n_pad, sm, bool(is_causal), s=sv_s)
    except Exception as e:
        if strict:
            raise
        _log_once("launch failed: %s" % e)
        return None, "launch_failed"
    if tensor_layout == "NHD":
        return out, None
    return out.view(B, H, N, D), None
