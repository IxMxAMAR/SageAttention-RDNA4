"""Prebuilt SK1 backend for the gfx12 native path (on by default on gfx1201).

This package ships prebuilt gfx1201 code objects (and, opt-in, the same set built for gfx1200) plus
the loader that launches them through the HIP runtime DLL that torch has already loaded. These code
objects are packaged:

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
* `sk1_t6i.gfx1201.hsaco` (contiguous HND) and `sk1_t6in.gfx1201.hsaco` (strided HND and NHD) are the
  INT8 Q.K variants. Q.K runs on `v_wmma_i32_16x16x16_iu8` with K quantised per token to int8; the
  V / P.V path is the fp8 one. They serve fp16, head_dim 128, and are used by default.
* `sk1_t4a1sb.gfx1201.hsaco` (contiguous HND) and `sk1_t4a1nb.gfx1201.hsaco` (strided HND and NHD)
  are the same kernels as `sk1_t4a1s` / `sk1_t4a1n` with a bf16 Q/O element type. They serve bf16,
  head_dim 128, and are used by default for bf16 input.
* `sk1_d64.gfx1201.hsaco` is the head_dim-64 port of the strided kernel. It is packaged but nothing
  routes to it: it was slower than the PR #368 kernel at head_dim 64, so those calls still use that
  path.
* `h34_prologue.gfx1201.hsaco` is the HIP KV prologue (`kernels/hip/h34_prologue.hip`). It quantises
  K and V for the attention objects above in at most two launches and is used by default; see
  `SAGEATTN_SK1_PROLOGUE` below.

`sk1_t4a1` is `sk1_t1` with the 32-element mask/skip select group placed behind one CTA-uniform
`if (need_mask)` branch. It is bit-identical to `sk1_t1` on the accuracy and edge-length test cells
and a few percent faster.

Why a prebuilt code object and not a compiled extension: a `.hsaco` is a self-contained
`elf64-amdgpu` file with no C++ imports, so the `c10` symbol ABI mismatch that breaks a `.pyd`
extension against a different torch build cannot arise. The loader is a plain `ctypes` call with
integer and pointer arguments.

## Default semantics

The backend is used only when all of these hold: the device arch is `gfx1201` (or `gfx1200` with
`SAGEATTN_SK1_GFX1200=1`), a HIP runtime resolves, the code object loads, and the call is inside
`ENVELOPE`. Any other case falls back to the
shipped path (the gfx12 native extension on gfx12, Triton on other arches), with at most one short
log line per process, and never raises at call time.

* `SAGEATTN_SK1_BACKEND=1` forces it on; a load failure then raises.
* `SAGEATTN_SK1_BACKEND=0` forces it off (the shipped path).
* Unset means on.
* `SAGEATTN_SK1_INT8`: unset serves fp16 head_dim-128 calls with the int8 objects and falls back to
  the fp8 objects, then to the shipped path, for a call they cannot serve. `=0` uses the fp8 objects
  only. `=1` forces int8 and is strict: a call int8 cannot serve raises.
* `SAGEATTN_SK1_BF16`: unset serves bf16 head_dim-128 calls with the bf16 objects and falls back
  normally. `=0` sends bf16 calls to the shipped path (reason `"dtype"`). `=1` forces the bf16
  objects and is strict.
* `SAGEATTN_SK1_BACKEND=0` wins over both.
* `SAGEATTN_SK1_PROLOGUE`: unset or `hip` quantises K and V with the HIP prologue and falls back to
  the Triton prologue for a call it cannot serve. `triton` uses the Triton prologue only. Only the
  prologue changes; the attention objects and their launch are the same.
* `SAGEATTN_SK1_GFX1200=1` lets a gfx1200 device use the `*_1200.hsaco` objects, which are the same
  kernels built with `--offload-arch=gfx1200`. Unset, a gfx1200 device is refused (reason `"arch"`)
  and takes the shipped path, as before. The switch does nothing on gfx1201: the device decides
  which object set is loaded.

The `.hsaco` files are built for gfx1201, plus the `_1200` set for gfx1200. There is no in-wheel
rebuild path; `tools/build_hsaco.py` in the source repository rebuilds them from `kernels/hip/`. The
arch check keeps any other device on the shipped path instead of raising.

The HIP runtime is resolved at call time and no install path is hard-coded. `rocm_bin_dir()` tries,
in order: `$SK1_HIP_DLL`, the `amdhip64*.dll` already loaded in this process (torch's own copy),
`$SK1_ROCM_BIN`, then the `_rocm_sdk_core` / `_rocm_sdk_devel` layouts under `site-packages`. The
`_rocm_sdk_core` copy is preferred because it is the one torch loads; `_rocm_sdk_devel` is a different
build, and loading it would put a second HIP runtime in the process (see `sk1_loader.py`).
"""
from __future__ import annotations

import os
import re
import sys

# `quant_kv_v8t` does `from quant_triton import ...` as a flat top-level import, so the package
# directory must be importable as a top-level location before it is imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = ["available", "try_sk1_t1", "SK1_HSACO_NAME", "ENVELOPE", "SUPPORTED_ARCHS",
           "device_arch", "arch_supported", "SK1_INT8", "SK1_INT8_STRICT", "int8_status",
           "SK1_BF16", "SK1_BF16_STRICT", "bf16_status",
           "SK1_GFX1200", "SK1_GFX1200_ARCHS", "gfx1200_status", "arch_suffix", "selected_arch"]

# ---- gfx1200 objects, opt-in ---------------------------------------------------------------------
#: Every gfx1201 code object is also built with `--offload-arch=gfx1200` (RX 9060 XT, the same RDNA4
#: instruction set) and shipped as `*_1200.hsaco`. The `.text` sections are byte-identical to the
#: gfx1201 ones. A gfx1200 device is routed to them only when this switch is set.
#:
#: `SAGEATTN_SK1_GFX1200=1` -> a gfx1200 device is served by the `*_1200.hsaco` objects.
#: unset / `0` / anything else -> a gfx1200 device is refused with `reason="arch"` and one log line
#: that names this switch. Read once, at import (no per-call override, no `os.environ` read on the hot
#: path), like `SAGEATTN_SK1_INT8`, `SAGEATTN_SK1_BF16` and `SAGEATTN_SK1_PROLOGUE`.
#:
#: The switch does nothing on gfx1201: a gfx1201 device always gets the `.gfx1201.hsaco` objects.
SK1_GFX1200_DEFAULT_ON = False
#: the architectures that have their own object set
SK1_GFX1200_ARCHS = ("gfx1200",)

_sk1_g12_env = os.environ.get("SAGEATTN_SK1_GFX1200", "").strip().lower()
if _sk1_g12_env in ("1", "true", "yes", "on"):
    SK1_GFX1200 = True
elif _sk1_g12_env in ("0", "false", "no", "off"):
    SK1_GFX1200 = False
else:
    SK1_GFX1200 = bool(SK1_GFX1200_DEFAULT_ON)

#: the arch the shipped gfx1201 object set is built for -- the default and the fallback
_DEFAULT_ARCH = "gfx1201"


def arch_suffix(arch):
    """The file-name suffix for `arch`'s object set. `""` for the gfx1201 set (so its file names are
    unchanged), `"_1200"` for the gfx1200 set."""
    return "" if arch == _DEFAULT_ARCH else "_" + arch.replace("gfx", "")


def _arch_of(name):
    """The inverse of `arch_suffix`: the arch tag a file name carries."""
    m = re.search(r"_(\d{4})\.hsaco$", name)
    return ("gfx" + m.group(1)) if m else _DEFAULT_ARCH


def selected_arch(device=None):
    """Which object set serves `device`, or `None` if none may.

    This is the one place the arch/switch decision is made; `arch_supported`, the object-name
    resolver and the log line all read it, so they cannot disagree.

    * `gfx1201` -> `"gfx1201"`, always (the switch has no effect).
    * `gfx1200` -> `"gfx1200"` iff `SK1_GFX1200`, else `None` (the arch refusal).
    * anything else, or an unreadable arch -> `None`.
    """
    arch = device_arch(device)
    if not arch:
        return None
    if any(arch.startswith(a) for a in SK1_GFX1200_ARCHS):
        return arch if SK1_GFX1200 else None
    return _DEFAULT_ARCH if arch.startswith(_DEFAULT_ARCH) else None


#: The arch whose file names the base constants below carry. This is always `gfx1201`, on purpose:
#: the base paths (`SK1_HSACO_PATH*`) are the literal gfx1201 file names, and `object_path()` rewrites
#: them by replacing `.gfx1201.hsaco`. If this followed the switch, a gfx1201 device with
#: `SAGEATTN_SK1_GFX1200=1` would get a `_1200` base path, `object_path` would find nothing to
#: replace, and the loader would hand a gfx1200 object to a gfx1201 device
#: (`hipModuleLoadData` rc=209, `hipErrorNoBinaryForGpu`). The switch only permits gfx1200; the device
#: chooses the object set, per call, through `_arch_now()` / `selected_arch()`.
_SEL_ARCH = _DEFAULT_ARCH
_SUF = arch_suffix(_SEL_ARCH)


def _arch_now(device=None):
    """The object set for this device right now: `selected_arch()`, or the gfx1201 default.

    Every loader and the prologue default to this, never to `_SEL_ARCH`, so the object set always
    follows the device and a gfx1200 object is never loaded on a gfx1201 device (or the reverse).
    """
    return selected_arch(device) or _DEFAULT_ARCH


def object_path(base_gfx1201_path, arch):
    """The object path for `arch`, given the gfx1201 path.

    This makes "a gfx1200 object is never loaded on gfx1201, or the reverse" structural: the suffix
    is computed from the same `arch` value the arch check used, so no code path produces a gfx1200
    name while the active arch is gfx1201. A gfx1200 object is `sk1_t4a1s_1200.hsaco`; the gfx1201
    object keeps its exact name.
    """
    suf = arch_suffix(arch)
    return base_gfx1201_path if not suf else \
        base_gfx1201_path.replace(".gfx1201.hsaco", "%s.hsaco" % suf)


def _with_suffix(name):
    """The module's own default-arch spelling of a gfx1201 object name."""
    return object_path(name, _SEL_ARCH)


#: The earlier, unfixed object. Kept for reference; nothing routes to it.
SK1_HSACO_NAME = _with_suffix("sk1_t4a1.gfx1201.hsaco")
SK1_HSACO_PATH = os.path.join(_HERE, SK1_HSACO_NAME)

#: The packaged object is `sk1_t4a1s` (12 arguments: `S` between `VMEAN` and `O`). The earlier
#: object `sk1_t4a1.gfx1201.hsaco` is kept in the package for reference and for A/B timing; nothing
#: routes to it.
SK1_PACKAGED_WITH_S = True
SK1_HSACO_NAME_S = _with_suffix("sk1_t4a1s.gfx1201.hsaco")
SK1_HSACO_PATH_S = os.path.join(_HERE, SK1_HSACO_NAME_S)
SK1_SYMBOLS_S = ("sk1t4a1s_attn_fwd_c0", "sk1t4a1s_attn_fwd_c1")

#: The widened object, `sk1_t4a1n`: the 12 arguments of `sk1_t4a1s` plus six `int` Q/O strides
#: (`kernels/hip/sk1_t4a1n.hip`, derived from `sk1_t4a1s.hip` by textual substitution). It addresses
#: contiguous HND exactly like `sk1_t4a1s`, but contiguous HND calls are still routed to
#: `sk1_t4a1s` so that path is unchanged. The widened object serves only strided HND and NHD calls.
SK1_HSACO_NAME_N = _with_suffix("sk1_t4a1n.gfx1201.hsaco")
SK1_HSACO_PATH_N = os.path.join(_HERE, SK1_HSACO_NAME_N)
SK1_SYMBOLS_N = ("sk1t4a1n_attn_fwd_c0", "sk1t4a1n_attn_fwd_c1")

#: The head_dim-64 object, `sk1_d64`: `sk1_t4a1n` with `HD 128 -> 64`, `KSTRIDE 152 -> 72` and 4 Q/output
#: fragments per wave instead of 8 (`kernels/hip/sk1_d64.hip`). It is the 18-argument strided form, so
#: it serves both contiguous-HND and physically-NHD D=64 calls, and it is the only object that serves
#: D=64.
#:
#: Three arms live in this one code object and differ only in the entry-point names:
#:   `sk1d64`  BM=128, 8 warps,  `amdgpu_waves_per_eu(6,6)`  (measured VGPR 224 -> 6 waves/SIMD32)
#:   `sk1d64w` BM=128, 8 warps,  `amdgpu_waves_per_eu(8,8)`  (measured VGPR 176 -> 8 waves/SIMD32)
#:   `sk1d64m` BM=256, 16 warps, `amdgpu_waves_per_eu(8,8)`  (measured VGPR 176 -> 8 waves/SIMD32)
#: `SK1_D64_ARM` is the arm the envelope routes to. It stays `None`, so nothing is routed and every D=64
#: call falls back to the shipped path: no arm beat the PR #368 kernel at head_dim 64. The default arm
#: also needed 224 registers where about 176 were expected, so its occupancy stayed at the D=128
#: kernel's.
SK1_HSACO_NAME_D64 = _with_suffix("sk1_d64.gfx1201.hsaco")
SK1_HSACO_PATH_D64 = os.path.join(_HERE, SK1_HSACO_NAME_D64)
SK1_D64_ARMS = {
    "sk1d64": ("sk1d64_attn_fwd_c0", "sk1d64_attn_fwd_c1"),
    "sk1d64w": ("sk1d64w_attn_fwd_c0", "sk1d64w_attn_fwd_c1"),
    "sk1d64m": ("sk1d64m_attn_fwd_c0", "sk1d64m_attn_fwd_c1"),
}
SK1_D64_ARM = None
SK1_SYMBOLS_D64 = SK1_D64_ARMS["sk1d64"]

#: The bf16 head_dim-64 object. `sk1_d64b` is `sk1_d64` with the Q/O element type as a template
#: parameter, the same recipe as the D=128 bf16 pair, so one source yields the fp16 arms and the bf16
#: arms. Every arm keeps its name and gains a `b` before the dtype suffix (`sk1d64bf`, `sk1d64wbf`,
#: `sk1d64mbf`); `sk1_loader.Sk1Attn` infers `qo_dtype`, `head_dim`, `bm`, `with_s` and `strided`
#: from those names, so a caller that only swaps `symbols` cannot feed a bf16 object fp16 Q or launch
#: it with the wrong argument count.
#:
#: `SK1_D64B_ARM` is the bf16 arm the envelope routes to. It stays `None`, so nothing is routed and
#: every bf16 D=64 call keeps the refusal `reason="bf16_d128_only"`. The object itself is not
#: packaged.
SK1_HSACO_NAME_D64B = _with_suffix("sk1_d64b.gfx1201.hsaco")
SK1_HSACO_PATH_D64B = os.path.join(_HERE, SK1_HSACO_NAME_D64B)
SK1_D64B_ARMS = {
    "sk1d64": ("sk1d64bf_attn_fwd_c0", "sk1d64bf_attn_fwd_c1"),
    "sk1d64w": ("sk1d64wbf_attn_fwd_c0", "sk1d64wbf_attn_fwd_c1"),
    "sk1d64m": ("sk1d64mbf_attn_fwd_c0", "sk1d64mbf_attn_fwd_c1"),
}
SK1_D64B_ARM = None

#: The INT8 Q.K objects, the default for fp16, D=128.
#:
#: `sk1_t6i`  (contiguous HND, `kernels/hip/sk1_t6i.hip`)
#: `sk1_t6in` (strided-HND / NHD: the same seven stride edits as `sk1_t4a1n`, applied to the int8
#:             source; bit-identical to `sk1_t6i` on contiguous HND)
#:
#: Q.K moves from fp8 e4m3 x fp8 e4m3 to int8 x int8 on `v_wmma_i32_16x16x16_iu8` with int32
#: accumulators; the V / P.V path is the fp8 one, unchanged. On the accuracy and edge-length test cells
#: the contiguous object was more accurate than the fp8 kernel on every cell, at a 1.6-2.5 %
#: kernel-only cost.
#:
#: The switch has three states, the same as the bf16 one:
#:
#:   * `SAGEATTN_SK1_INT8` unset  -> `SK1_INT8 = SK1_INT8_DEFAULT_ON` (True), `STRICT = False`:
#:     int8 serves every fp16 D=128 call it can, and falls back normally (fp8 SK1 first, then the
#:     PR #368 gfx12 native path) for every call it cannot;
#:   * `=0/false/no/off`          -> OFF, non-strict: the fp8 behaviour, bit for bit;
#:   * `=1/true/yes/on`           -> ON and strict: a call int8 cannot serve raises instead of falling
#:     back. This is the only mode in which a refusal is an error.
#:
#: D=128 only, fp16 only. Both objects were generated from the D=128 `sk1_t4a1s` (fp16 Q/O), so a
#: `head_dim == 64` call is a clean refusal (`reason="int8_d128_only"`) and a bf16 call never reaches
#: this path at all: bf16 takes precedence in the routing and is served by the
#: `sk1_t4a1sb` / `sk1_t4a1nb` pair. There is no int8 bf16 object.
SK1_HSACO_NAME_I8 = _with_suffix("sk1_t6i.gfx1201.hsaco")
SK1_HSACO_PATH_I8 = os.path.join(_HERE, SK1_HSACO_NAME_I8)
SK1_SYMBOLS_I8 = ("sk1t6i_attn_fwd_c0", "sk1t6i_attn_fwd_c1")
SK1_HSACO_NAME_I8N = _with_suffix("sk1_t6in.gfx1201.hsaco")
SK1_HSACO_PATH_I8N = os.path.join(_HERE, SK1_HSACO_NAME_I8N)
SK1_SYMBOLS_I8N = ("sk1t6in_attn_fwd_c0", "sk1t6in_attn_fwd_c1")
#: The single default switch for int8, as `SK1_BF16_DEFAULT_ON` is for bf16. `SAGEATTN_SK1_INT8=0`
#: restores the fp8 behaviour bit for bit.
SK1_INT8_DEFAULT_ON = True

#: `SAGEATTN_SK1_INT8`: `1/true/yes/on` -> ON and strict (a call the int8 path cannot serve raises);
#: `0/false/no/off` -> OFF (the fp8 behaviour); unset -> `SK1_INT8_DEFAULT_ON`, non-strict. Read once,
#: at import. `SAGEATTN_SK1_BACKEND=0` wins: no SK1 at all, int8 included.
_sk1_i8_env = os.environ.get("SAGEATTN_SK1_INT8", "").strip().lower()
if _sk1_i8_env in ("1", "true", "yes", "on"):
    SK1_INT8 = True
    SK1_INT8_STRICT = True
elif _sk1_i8_env in ("0", "false", "no", "off"):
    SK1_INT8 = False
    SK1_INT8_STRICT = False
else:
    SK1_INT8 = bool(SK1_INT8_DEFAULT_ON)
    SK1_INT8_STRICT = False

#: The bf16 objects. `sk1_t4a1sb` (contiguous HND, 12-argument, `__bf16` Q/O) and `sk1_t4a1nb`
#: (strided-HND / NHD, 18-argument). Both are the same kernel source as the fp16 pair, with the Q/O
#: element type as a template parameter (`kernels/hip/sk1_t4a1sb.hip` / `kernels/hip/sk1_t4a1nb.hip`).
#: Each object also contains an inert `_Float16` instantiation (`sk1t4a1sp_*` / `sk1t4a1np_*`), kept so
#: the fp16 code in the templated source can be compared instruction for instruction against the
#: shipped fp16 object; it had no differing lines.
#:
#: The dtype contract is per object (`sk1_loader.Sk1Attn.qo_dtype`): the fp16 objects refuse a bf16 Q
#: and these refuse an fp16 one.
#:
#: `SK1_BF16_DEFAULT_ON` is the single default switch and it is `True`: an unset `SAGEATTN_SK1_BF16`
#: serves a bf16 call with the bf16 objects and falls back normally (fp8 SK1 first, then PR #368) for a
#: call they cannot serve. `=1` forces ON and strict, `=0` forces OFF (a bf16 call then returns reason
#: `"dtype"`).
SK1_HSACO_NAME_B = _with_suffix("sk1_t4a1sb.gfx1201.hsaco")
SK1_HSACO_PATH_B = os.path.join(_HERE, SK1_HSACO_NAME_B)
SK1_SYMBOLS_B = ("sk1t4a1sb_attn_fwd_c0", "sk1t4a1sb_attn_fwd_c1")
SK1_HSACO_NAME_BN = _with_suffix("sk1_t4a1nb.gfx1201.hsaco")
SK1_HSACO_PATH_BN = os.path.join(_HERE, SK1_HSACO_NAME_BN)
SK1_SYMBOLS_BN = ("sk1t4a1nb_attn_fwd_c0", "sk1t4a1nb_attn_fwd_c1")
SK1_BF16_DEFAULT_ON = True

_sk1_b_env = os.environ.get("SAGEATTN_SK1_BF16", "").strip().lower()
if _sk1_b_env in ("1", "true", "yes", "on"):
    SK1_BF16 = True
    SK1_BF16_STRICT = True
elif _sk1_b_env in ("0", "false", "no", "off"):
    SK1_BF16 = False
    SK1_BF16_STRICT = False
else:
    SK1_BF16 = bool(SK1_BF16_DEFAULT_ON)
    SK1_BF16_STRICT = False

#: The arches a packaged code object exists for. The comparison is a prefix test, so a
#: `gcnArchName` such as `gfx1201:sramecc-:xnack-` also matches.
#:
#: Membership means "an object set exists", not "this device is served". The serve decision is
#: `selected_arch()`: a gfx1200 device is served only when `SAGEATTN_SK1_GFX1200` is set, so with the
#: switch unset it takes the arch refusal (`reason="arch"`) with one log line naming the switch.
SUPPORTED_ARCHS = ("gfx1201", "gfx1200")

#: The exact envelope `try_sk1_t1` serves.  Anything else returns `None` (caller falls back).
#: `layout` is HND or NHD, and `contiguous` means "contiguous in its own layout, or a physically-NHD
#: view" (`sk1_loader._strided_triple`). Contiguous HND is served by `sk1_t4a1s`; everything else by
#: `sk1_t4a1n`. `dtype` is fp16 or bf16; a bf16 call is served by the `sk1_t4a1sb` / `sk1_t4a1nb` pair
#: (the fp16 sources with the Q/O type templated) when `SK1_BF16` allows it, and otherwise falls back
#: with reason `"dtype"`. `head_dim` 64 is inside the envelope, but no D=64 object is routed (see
#: `SK1_D64_ARM`), so those calls fall back.
ENVELOPE = dict(
    layout=("HND", "NHD"),
    dtype=("fp16", "bf16"),
    head_dim=(64, 128),
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
_BACKEND_D64 = None
_BACKEND_D64_FAILED = None
_BACKEND_D64B = None
_BACKEND_D64B_FAILED = None
_BACKEND_I8 = None
_BACKEND_I8_FAILED = None
_BACKEND_I8N = None
_BACKEND_I8N_FAILED = None
#: The bf16 pair.
_BACKEND_B = None
_BACKEND_B_FAILED = None
_BACKEND_BN = None
_BACKEND_BN_FAILED = None
_LOGGED = False
#: `sk1_loader.strided_view_reason`, resolved once on first use. The stride guard is on the hot path
#: of every call (`_envelope_ok` runs three times per attention call) and the non-kernel part of the
#: path measures about 0.20 ms per call, so the import is cached rather than repeated.
_SVR = None


def _strided_reason(t, name):
    """The widened-acceptance guard. Returns `None` if `t` is servable, else a reason."""
    global _SVR
    if _SVR is None:
        from .sk1_loader import strided_view_reason
        _SVR = strided_view_reason
    return _SVR(t, name)


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
    """True iff `device` may be served by a packaged code object right now.

    This is `selected_arch(device) is not None`: membership in `SUPPORTED_ARCHS` and the gfx1200
    switch. A gfx1200 device with `SAGEATTN_SK1_GFX1200` unset is therefore unsupported.

    The module attribute is consulted through `globals()` so that a test which replaces this function
    (for example with `lambda *a, **kw: False`, to simulate another device) is honoured; a plain call
    would bypass the patch. A replacement that is not a plain function (such as the `MagicMock`
    that `unittest.mock.patch` injects) is called as the patched callable.
    """
    import types
    cur = globals().get("arch_supported")
    if not isinstance(cur, types.FunctionType):
        return bool(cur(device))
    return selected_arch(device) is not None


def _arch_refusal_reason(device=None) -> str:
    """The one-line reason for an arch refusal, naming the switch when that is what blocked it."""
    arch = device_arch(device) or "<unknown>"
    if arch and any(arch.startswith(a) for a in SK1_GFX1200_ARCHS) and not SK1_GFX1200:
        return ("device arch %r needs its own objects; set SAGEATTN_SK1_GFX1200=1 to enable them "
                "(they are opt-in until a gfx1200 tester confirms them)" % arch)
    return "device arch %r is not %s" % (arch, _DEFAULT_ARCH)


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


# ---- the HIP KV prologue router ------------------------------------------------------------------
#: `SAGEATTN_SK1_PROLOGUE` picks the prologue that runs before the attention object: `triton` (the
#: shipped `quant_kv_v8t*` kernels) or `hip` (`kernels/hip/h34_prologue.hip`, built to
#: `h34_prologue.gfx1201.hsaco`, loaded and launched through `sk1_loader.HipRuntime`). Read once, at
#: import. Unset -> `SK1_PROLOGUE_DEFAULT`.
#:
#: The default is `"hip"`, so the default D=128 path compiles and launches no Triton kernels.
#: `SAGEATTN_SK1_PROLOGUE=triton` restores the Triton prologue exactly; any other value logs nothing
#: and means the default. A call the HIP prologue cannot serve falls back to the Triton prologue
#: normally, never to the gfx12 native path, and only the prologue moves: the attention objects and
#: their launch are untouched.
SK1_PROLOGUE_DEFAULT = "hip"
_sk1_prol_env = os.environ.get("SAGEATTN_SK1_PROLOGUE", "").strip().lower()
if _sk1_prol_env in ("hip", "triton"):
    SK1_PROLOGUE = _sk1_prol_env
else:
    SK1_PROLOGUE = SK1_PROLOGUE_DEFAULT

_HIP_P = None
_HIP_P_TRIED = False
#: the arch the loaded prologue was built for (see `_hip_prologue`)
_HIP_P_ARCH = None
_H34_LOGGED = 0


def _log_h34(msg: str) -> None:
    """The prologue's own at-most-8-lines logger. Deliberately not `_log_once`, which owns the one
    "SK1 backend not used" line and must stay available for a genuine fallback."""
    global _H34_LOGGED
    if _H34_LOGGED >= 8:
        return
    _H34_LOGGED += 1
    try:
        sys.stderr.write("[sageattention] HIP prologue: %s\n" % msg)
        sys.stderr.flush()
    except Exception:                                          # pragma: no cover
        pass


def _hip_prologue(arch=None):
    """The loaded HIP prologue object, or `None`. Loaded once per process; never raises.

    The prologue is a code object too, so it is per-arch: `h34_prologue.gfx1201.hsaco` /
    `h34_prologue_1200.hsaco`. `arch` selects it (default: this device's set, `_arch_now()`), and
    the loaded instance is keyed on it like every `_load_*`, so a gfx1201 prologue can never serve a
    gfx1200 call.
    """
    global _HIP_P, _HIP_P_TRIED, _HIP_P_ARCH
    arch = arch or _arch_now()
    if _HIP_P_TRIED and _HIP_P_ARCH == arch:
        return _HIP_P
    _HIP_P_TRIED = True
    _HIP_P_ARCH = arch
    _HIP_P = None
    if SK1_PROLOGUE != "hip":
        return None
    try:
        from .hip_prologue import HipPrologue, hsaco_path_for
        _HIP_P = HipPrologue(hsaco_path=hsaco_path_for(arch))
    except Exception as e:                                      # noqa: BLE001
        _log_h34("the HIP prologue did not load (%s); the Triton prologue serves the calls" % e)
        _HIP_P = None
    return _HIP_P


def packaged_hsaco_path(arch=None) -> str:
    """The object the packaged path loads: `sk1_t4a1s`, or `sk1_t4a1` when `SK1_PACKAGED_WITH_S` is False.

    `arch` selects the object set; it defaults to the device's set (`_arch_now()`), never to the
    import-time switch value, so a gfx1200 name cannot appear on a gfx1201 device.
    """
    return object_path(SK1_HSACO_PATH_S if SK1_PACKAGED_WITH_S else SK1_HSACO_PATH,
                       arch or _arch_now())


def gfx1200_status() -> dict:
    """The state of the gfx1200 opt-in without loading anything and without a device. Never raises.

    `{"switch": SAGEATTN_SK1_GFX1200, "default_on": False, "archs": ("gfx1200",),
      "supported_archs": (...), "objects": {name: {"file", "present", "sha256", "size"}},
      "gfx1201_objects": {...}}`.
    """
    def one(path):
        out = dict(file=os.path.basename(path), path=path.replace("\\", "/"),
                   present=os.path.isfile(path), sha256=None, size=None)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    names = ("sk1_t4a1s", "sk1_t4a1n", "sk1_t4a1sb", "sk1_t4a1nb", "sk1_t6i", "sk1_t6in",
             "sk1_t4a1", "h34_prologue")
    return dict(switch=bool(SK1_GFX1200), default_on=bool(SK1_GFX1200_DEFAULT_ON),
                archs=list(SK1_GFX1200_ARCHS), supported_archs=list(SUPPORTED_ARCHS),
                base_arch=_SEL_ARCH, selected_now=_arch_now(),
                objects={n: one(object_path(os.path.join(_HERE, n + ".gfx1201.hsaco"), "gfx1200"))
                         for n in names},
                gfx1201_objects={n: one(os.path.join(_HERE, n + ".gfx1201.hsaco"))
                                 for n in names})


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


#: The arch the currently cached objects were loaded for. A cache built for one arch must never
#: serve a call routed for the other, so every `_load_*` is keyed on it and a mismatch drops the
#: caches. On a real device the arch is constant for the process, so this costs one comparison.
_ARCH_LOADED = None

#: Every loader cache and its "already failed" flag. An arch change must drop all of them, not just
#: the one being loaded: a call can load the int8 object and then fall through to the fp8 object, so
#: dropping only `_BACKEND` would leave `_BACKEND_I8` holding a module built for the previous arch.
#: That cannot happen on a real device, where the arch never changes, but it can when the arch probe
#: is monkeypatched.
_ARCH_CACHES = (
    "_BACKEND", "_BACKEND_FAILED",
    "_BACKEND_N", "_BACKEND_N_FAILED",
    "_BACKEND_D64", "_BACKEND_D64_FAILED",
    "_BACKEND_D64B", "_BACKEND_D64B_FAILED",
    "_BACKEND_I8", "_BACKEND_I8_FAILED",
    "_BACKEND_I8N", "_BACKEND_I8N_FAILED",
    "_BACKEND_B", "_BACKEND_B_FAILED",
    "_BACKEND_BN", "_BACKEND_BN_FAILED",
)


def _cache_arch(arch, cache_name, failed_name):
    """Drop every loader cache if it was built for a different arch, and record the arch.

    This completes the loader's per-arch containment: even if a caller monkeypatches the arch probe
    between calls, no loader can hand back an object whose file name carries the other arch's
    suffix. The HIP prologue cache is keyed on its own `_HIP_P_ARCH` and drops itself, but it is
    reset here too so the two cannot disagree.
    """
    global _ARCH_LOADED, _HIP_P, _HIP_P_TRIED, _HIP_P_ARCH
    if _ARCH_LOADED != arch:
        for name in _ARCH_CACHES:
            globals()[name] = None
        _HIP_P = None
        _HIP_P_TRIED = False
        _HIP_P_ARCH = None
        _ARCH_LOADED = arch
    return arch


def _load(arch=None):
    global _BACKEND, _BACKEND_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND", "_BACKEND_FAILED")
    if _BACKEND is not None:
        return _BACKEND
    suf = arch_suffix(arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        rt = HipRuntime()
        if SK1_PACKAGED_WITH_S:
            p = object_path(SK1_HSACO_PATH_S, arch)
            _BACKEND = Sk1Attn(p, rt, symbols=SK1_SYMBOLS_S, with_s=True)
        else:
            p = object_path(SK1_HSACO_PATH, arch)
            _BACKEND = Sk1Attn(p, rt)
        return _BACKEND
    except Exception as e:
        _BACKEND_FAILED = "%s: %s" % (type(e).__name__, e)
        raise


def _load_n(arch=None):
    """The widened object `sk1_t4a1n`, or `None` if it is not installed or will not load.

    Kept separate from `_load()` so that code which swaps `SK1_PACKAGED_WITH_S` and `_BACKEND` to
    select a particular object keeps working. A missing widened object is a clean refusal
    (`reason="no_strided_object"`), never a raise on the default path.

    `arch` selects the object set; it defaults to this device's set (`_arch_now()`), never to the
    import-time switch value, so the object set always follows the device.
    """
    global _BACKEND_N, _BACKEND_N_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_N", "_BACKEND_N_FAILED")
    if _BACKEND_N is not None:
        return _BACKEND_N
    if _BACKEND_N_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_N, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_N_FAILED = "missing %s" % p
            return None
        _BACKEND_N = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_N,
                             with_s=True, strided=True)
        return _BACKEND_N
    except Exception as e:                                   # pragma: no cover
        _BACKEND_N_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_d64():
    """The head_dim-64 object, or `None` if it is not wired, not installed or will not load.

    Returns `None` (a clean refusal, `reason="no_d64_object"`, never a raise) in three cases:
    `SK1_D64_ARM` is `None`, the `.hsaco` is missing, or it will not load. Same shape as `_load_n()`
    so that a partial install degrades to the shipped path.
    """
    global _BACKEND_D64, _BACKEND_D64_FAILED
    if SK1_D64_ARM is None:
        _BACKEND_D64_FAILED = "no arm selected (SK1_D64_ARM is None)"
        return None
    if _BACKEND_D64 is not None:
        return _BACKEND_D64
    if _BACKEND_D64_FAILED is not None and _BACKEND_D64_FAILED != "no arm selected (SK1_D64_ARM is None)":
        return None
    if SK1_D64_ARM not in SK1_D64_ARMS:
        _BACKEND_D64_FAILED = "unknown arm %r" % (SK1_D64_ARM,)
        return None
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(SK1_HSACO_PATH_D64):
            _BACKEND_D64_FAILED = "missing %s" % SK1_HSACO_PATH_D64
            return None
        syms = SK1_D64_ARMS[SK1_D64_ARM]
        _BACKEND_D64 = Sk1Attn(SK1_HSACO_PATH_D64, HipRuntime(), symbols=syms,
                               with_s=True, strided=True, head_dim=64)
        _BACKEND_D64_FAILED = None
        return _BACKEND_D64
    except Exception as e:                                   # pragma: no cover
        _BACKEND_D64_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_d64b():
    """The head_dim-64 bf16 object, or `None` if it is not wired, not installed or will not load.

    The same shape as `_load_d64()`: `SK1_D64B_ARM is None` (nothing routed), a missing `.hsaco`, or
    a load failure all return `None`, a clean refusal (`reason="no_d64b_object"`), never a raise. On
    the default path this is not called at all, because `SK1_D64B_ARM` is `None`, so every bf16 D=64
    call keeps the refusal `"bf16_d128_only"`.
    """
    global _BACKEND_D64B, _BACKEND_D64B_FAILED
    if SK1_D64B_ARM is None:
        _BACKEND_D64B_FAILED = "no arm selected (SK1_D64B_ARM is None)"
        return None
    if _BACKEND_D64B is not None:
        return _BACKEND_D64B
    if _BACKEND_D64B_FAILED is not None and \
            _BACKEND_D64B_FAILED != "no arm selected (SK1_D64B_ARM is None)":
        return None
    if SK1_D64B_ARM not in SK1_D64B_ARMS:
        _BACKEND_D64B_FAILED = "unknown arm %r" % (SK1_D64B_ARM,)
        return None
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(SK1_HSACO_PATH_D64B):
            _BACKEND_D64B_FAILED = "missing %s" % SK1_HSACO_PATH_D64B
            return None
        syms = SK1_D64B_ARMS[SK1_D64B_ARM]
        _BACKEND_D64B = Sk1Attn(SK1_HSACO_PATH_D64B, HipRuntime(), symbols=syms,
                                with_s=True, strided=True, head_dim=64)
        _BACKEND_D64B_FAILED = None
        return _BACKEND_D64B
    except Exception as e:                                   # pragma: no cover
        _BACKEND_D64B_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_i8(arch=None):
    """The contiguous-HND INT8 object `sk1_t6i`, or `None` if it is not installed or will not load.

    Same shape as `_load_n()`: a missing object is a clean refusal (`reason="no_int8_object"`), never
    a raise on the default path.

    `arch` selects the object set (default: the device's).
    """
    global _BACKEND_I8, _BACKEND_I8_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_I8", "_BACKEND_I8_FAILED")
    if _BACKEND_I8 is not None:
        return _BACKEND_I8
    if _BACKEND_I8_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_I8, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_I8_FAILED = "missing %s" % p
            return None
        _BACKEND_I8 = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_I8, with_s=True)
        return _BACKEND_I8
    except Exception as e:                                   # pragma: no cover
        _BACKEND_I8_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_i8n(arch=None):
    """The strided-HND / NHD INT8 object `sk1_t6in`, or `None`.  See `_load_i8()`."""
    global _BACKEND_I8N, _BACKEND_I8N_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_I8N", "_BACKEND_I8N_FAILED")
    if _BACKEND_I8N is not None:
        return _BACKEND_I8N
    if _BACKEND_I8N_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_I8N, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_I8N_FAILED = "missing %s" % p
            return None
        _BACKEND_I8N = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_I8N,
                               with_s=True, strided=True)
        return _BACKEND_I8N
    except Exception as e:                                   # pragma: no cover
        _BACKEND_I8N_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def int8_status() -> dict:
    """The state of the two int8 objects without loading anything.  Never raises.

    `{"enabled": SK1_INT8, "strict": SK1_INT8_STRICT, "contig": {...}, "strided": {...}}` where each
    inner dict has `file`, `present`, `sha256`, `size`, `loaded`, `failed`.  Useful for recording which
    object served a run.
    """
    def one(path, name, loaded, failed):
        out = dict(file=name, present=os.path.isfile(path), sha256=None, size=None,
                   loaded=(loaded is not None), failed=failed)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    return dict(enabled=bool(SK1_INT8), strict=bool(SK1_INT8_STRICT),
                default_on=bool(SK1_INT8_DEFAULT_ON),
                contig=one(SK1_HSACO_PATH_I8, SK1_HSACO_NAME_I8, _BACKEND_I8, _BACKEND_I8_FAILED),
                strided=one(SK1_HSACO_PATH_I8N, SK1_HSACO_NAME_I8N, _BACKEND_I8N,
                            _BACKEND_I8N_FAILED))


def _load_b(arch=None):
    """The bf16 object `sk1_t4a1sb` (contiguous HND, 12-argument), or `None` if unavailable.

    Separate from `_load()` for the same reason `_load_n()` is: code that swaps
    `SK1_PACKAGED_WITH_S` / `_BACKEND` to select a particular object must keep selecting it. A missing
    object is a clean refusal (`reason="no_bf16_object"`), never a raise on the default path.
    """
    global _BACKEND_B, _BACKEND_B_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_B", "_BACKEND_B_FAILED")
    if _BACKEND_B is not None:
        return _BACKEND_B
    if _BACKEND_B_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_B, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_B_FAILED = "missing %s" % p
            return None
        import torch
        _BACKEND_B = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_B,
                             with_s=True, strided=False, head_dim=128, qo_dtype=torch.bfloat16)
        return _BACKEND_B
    except Exception as e:                                   # pragma: no cover
        _BACKEND_B_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def _load_bn(arch=None):
    """The bf16 object `sk1_t4a1nb` (strided-HND / NHD, 18-argument), or `None`.

    Bit-identical to `sk1_t4a1sb` on contiguous HND with `(rs,hs,bs) = (D, N*D, H*N*D)`; the
    contiguous cell keeps the plain 12-argument object, exactly as the fp16 pair does.
    """
    global _BACKEND_BN, _BACKEND_BN_FAILED
    arch = _cache_arch(arch or _arch_now(), "_BACKEND_BN", "_BACKEND_BN_FAILED")
    if _BACKEND_BN is not None:
        return _BACKEND_BN
    if _BACKEND_BN_FAILED is not None:
        return None
    p = object_path(SK1_HSACO_PATH_BN, arch)
    try:
        from .sk1_loader import HipRuntime, Sk1Attn
        if not os.path.isfile(p):
            _BACKEND_BN_FAILED = "missing %s" % p
            return None
        import torch
        _BACKEND_BN = Sk1Attn(p, HipRuntime(), symbols=SK1_SYMBOLS_BN,
                              with_s=True, strided=True, head_dim=128, qo_dtype=torch.bfloat16)
        return _BACKEND_BN
    except Exception as e:                                   # pragma: no cover
        _BACKEND_BN_FAILED = "%s: %s" % (type(e).__name__, e)
        return None


def bf16_status() -> dict:
    """The state of the two bf16 objects without loading anything.  Never raises.

    `{"enabled": SK1_BF16, "strict": SK1_BF16_STRICT, "default_on": SK1_BF16_DEFAULT_ON,
    "contig": {...}, "strided": {...}}`, each inner dict `file`, `present`, `sha256`, `size`,
    `loaded`, `failed`.  Useful for recording which object served a run.
    """
    def one(path, name, loaded, failed):
        out = dict(file=name, present=os.path.isfile(path), sha256=None, size=None,
                   loaded=(loaded is not None), failed=failed)
        try:
            with open(path, "rb") as fh:
                import hashlib
                h = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                out["sha256"] = h.hexdigest()
            out["size"] = os.path.getsize(path)
        except Exception:
            pass
        return out
    return dict(enabled=bool(SK1_BF16), strict=bool(SK1_BF16_STRICT),
                default_on=bool(SK1_BF16_DEFAULT_ON),
                contig=one(SK1_HSACO_PATH_B, SK1_HSACO_NAME_B, _BACKEND_B, _BACKEND_B_FAILED),
                strided=one(SK1_HSACO_PATH_BN, SK1_HSACO_NAME_BN, _BACKEND_BN,
                            _BACKEND_BN_FAILED))


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
    # The accepted dtypes are fp16 and bf16, and they must agree across q/k/v. The two have the same
    # 16-bit width but different exponent/mantissa splits, so a mixed triple would be read with the
    # wrong interpretation. This is a hard contract, not a switch.
    if not (q.dtype == k.dtype == v.dtype):
        return "dtype"
    if q.dtype not in (torch.float16, torch.bfloat16):
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
    # Normalise to the (B, H, N, D) reading the kernel uses, then apply the stride rule.  The rule
    # lives in one place -- `sk1_loader.strided_view_reason` -- so this envelope can never accept a
    # pattern the loader's own contract check would refuse, or the reverse.  Contiguous HND still
    # satisfies it trivially and still takes the unchanged object and code path below.
    for nm, t in (("q", qq), ("k", kk), ("v", vv)):
        r = _strided_reason(t, nm)
        if r is not None:
            return r
    q, k, v = qq, kk, vv
    # head_dim is 64 or 128, and the three tensors must agree on it (`q.size(3)` alone would let a D=64
    # q against a D=128 k reach the raw launch). `try_sk1_t1` picks the object from this same value.
    if q.size(3) not in (64, 128) or k.size(3) != q.size(3) or v.size(3) != q.size(3):
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
               smooth_k=True, smooth_v=False, attn_mask=None, strict=False, int8=None, bf16=None):
    """Run the prebuilt SK1 kernel if the call is inside `ENVELOPE` and the platform supports it.

    Returns `(out, reason)` where `out` is the attention output in the caller's (B,H,N,D) shape, or
    `(None, reason)` when the call is not served and the caller must fall back to the shipped path.

    `strict=False` (the default): a non-`gfx1201` arch, an unresolvable HIP
    runtime, a code object that will not load, a failure inside the fused prologue, or a launch
    failure **falls back** with at most one short log line per process and **never raises**.
    `strict=True` (`SAGEATTN_SK1_BACKEND=1`, or `sk1_backend=True`): the same failures **raise**,
    because silently falling back on a kernel that was explicitly demanded would hide a defect.

    `int8=None` reads the module constant `SK1_INT8` (`SAGEATTN_SK1_INT8`; `SK1_INT8_DEFAULT_ON` is
    the default switch and is `True`); `int8=True/False` forces it for this call.  With `int8` on, the
    two int8 objects serve the call (`sk1_t6i` for contiguous HND, `sk1_t6in` for strided-HND/NHD).

    The int8 objects are fp16-Q/O and head_dim 128 only, so with `int8` on and not strict (the
    default) a call they cannot serve falls back normally: first to the fp8 SK1 objects, and if those
    cannot serve it either, this function returns `(None, <the fp8 route's own reason>)` and the
    caller continues to its PR #368 path.  That reason is the fp8 route's (`"no_d64_object"` for
    `head_dim == 64`, `"seq_len"`/`"dtype"`/... for an envelope miss), never an int8-specific string.
    A bf16 call never reaches the int8 path at all: bf16 takes precedence and is served by the
    `sk1_t4a1sb`/`sk1_t4a1nb` pair.  `int8=True` (i.e. `SAGEATTN_SK1_INT8=1`, or `sk1_int8=True`) is
    strict for the int8 path: the same call raises.

    `bf16=None` reads the module constant `SK1_BF16` (`SAGEATTN_SK1_BF16`; `SK1_BF16_DEFAULT_ON` is
    the default switch); `bf16=True/False` forces it for this call.  A bf16 call with the bf16 objects
    switched off is a clean refusal `reason="dtype"`.  `head_dim == 64` is a clean refusal
    `bf16_d128_only` (both bf16 objects are D=128, like the int8 pair).
    """
    reason = _envelope_ok(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k,
                          smooth_v, attn_mask)
    if reason is not None:
        return None, reason
    # The arch check is the one place the arch/switch decision is made. `arch_supported(q.device)` is
    # called explicitly, as a module-attribute call, so a test that replaces that function to
    # simulate another device is honoured; reading `selected_arch()` alone would ignore it.
    # `arch_supported` is `selected_arch(...) is not None` unless it has been replaced.
    if not arch_supported(q.device):
        if strict:
            raise RuntimeError(
                "SK1 backend forced on (SAGEATTN_SK1_BACKEND=1 / sk1_backend=True) but the device "
                "arch is %r; the packaged code objects support %s (gfx1200 needs "
                "SAGEATTN_SK1_GFX1200=1)"
                % (device_arch(q.device) or "<unknown>", ", ".join(SUPPORTED_ARCHS)))
        _log_once(_arch_refusal_reason(q.device))
        return None, "arch"
    # The object set this call uses, derived from the same decision the check above approved, so a
    # gfx1200 object is never loaded on gfx1201 (or the reverse): the name carries the arch.
    _arch = selected_arch(q.device) or _DEFAULT_ARCH
    # Routing. Contiguous HND (all three tensors) goes to `sk1_t4a1s` on the plain path below.
    # Everything else (strided HND, NHD) goes to the widened `sk1_t4a1n`.
    if tensor_layout == "NHD":
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        qq, kk, vv = q, k, v
    contiguous_hnd = bool(tensor_layout == "HND" and q.is_contiguous() and k.is_contiguous()
                          and v.is_contiguous())
    # head_dim 64 is a different code object, and the same object for HND and NHD (it is the
    # 18-argument strided form). head_dim 128 keeps the routing above. While `SK1_D64_ARM` is None
    # nothing is routed and every D=64 call falls through to the shipped path.
    d64 = int(qq.size(3)) == 64
    # The dtype decides. bf16 has its own object pair (the int8 and fp16 pairs are fp16 Q/O only), so
    # bf16 takes precedence over int8. `SK1_BF16` is the switch; with it off a bf16 call returns
    # `"dtype"`.
    import torch
    is_bf16 = bool(q.dtype == torch.bfloat16)
    if bf16 is None:
        want_b, b_strict = bool(SK1_BF16), bool(SK1_BF16_STRICT)
    else:
        want_b, b_strict = bool(bf16), bool(bf16)
    if is_bf16 and not want_b:
        if b_strict or strict:
            raise RuntimeError(
                "bf16 SK1 was demanded (SAGEATTN_SK1_BF16=1 / sk1_bf16=True) but the bf16 objects "
                "are switched off; set SAGEATTN_SK1_BF16=1")
        _log_once("bf16 input but the bf16 objects are switched off")
        return None, "dtype"
    if is_bf16 and d64:
        # The bf16 D=128 objects were generated from the D=128 sources; serving a D=64 call with one
        # would read the wrong KSTRIDE. There is a separate D=64 bf16 object (`sk1_d64b`), routed to
        # only once `SK1_D64B_ARM` is set. Until then this is a clean refusal, the same shape as the
        # int8 one below.
        if SK1_D64B_ARM is None:
            if b_strict or strict:
                raise RuntimeError(
                    "the bf16 SK1 objects (%s / %s) are head_dim 128 only; head_dim is 64"
                    % (SK1_HSACO_NAME_B, SK1_HSACO_NAME_BN))
            _log_once("bf16 object is head_dim 128 only (head_dim=64)")
            return None, "bf16_d128_only"
    # The int8 objects, only when the switch says so.
    if int8 is None:
        want_i8, i8_strict = bool(SK1_INT8), bool(SK1_INT8_STRICT)
    else:
        want_i8, i8_strict = bool(int8), bool(int8)

    # The normal fallback. The int8 objects are D=128 / fp16 only. With `int8` on but not strict (the
    # default, `SAGEATTN_SK1_INT8` unset), a call the int8 path cannot serve falls back first to the
    # fp8 SK1 objects if they can serve it, else to the caller's PR #368 path. `_i8_fallback` is that
    # first step: it re-enters this function with `int8=False` and returns that call's
    # `(out, reason)` as is, so a D != 128 call comes back with `"no_d64_object"`, not an
    # int8-specific string. `int8=False` cannot re-enter this branch, so the recursion is one level
    # deep.
    #
    # `strict` never reaches here: every refusal below raises first when `strict` is set.
    def _i8_fallback():
        return try_sk1_t1(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
                          sm_scale=sm_scale, return_lse=return_lse, smooth_k=smooth_k,
                          smooth_v=smooth_v, attn_mask=attn_mask, strict=strict, int8=False,
                          bf16=bf16)

    if want_i8 and d64 and not is_bf16:
        # Both int8 objects were generated from the D=128 `sk1_t4a1s`; serving a D=64 call with one
        # would read the wrong KSTRIDE. Clean refusal, same shape as every other envelope miss.
        if i8_strict or strict:
            raise RuntimeError(
                "the int8 SK1 objects (%s / %s) are head_dim 128 only; head_dim is 64"
                % (SK1_HSACO_NAME_I8, SK1_HSACO_NAME_I8N))
        _log_once("int8 object is head_dim 128 only (head_dim=64)")
        return _i8_fallback()
    try:
        # Every loader is keyed on the arch the check above approved. The `d64` objects stay on the
        # module's own arch (nothing routes to them: `SK1_D64_ARM` / `SK1_D64B_ARM` are None).
        if is_bf16:
            attn = (_load_d64b() if d64 else
                    (_load_b(_arch) if contiguous_hnd else _load_bn(_arch)))
        elif want_i8:
            attn = _load_i8(_arch) if contiguous_hnd else _load_i8n(_arch)
        else:
            attn = (_load_d64() if d64 else
                    (_load(_arch) if contiguous_hnd else _load_n(_arch)))
    except Exception as e:
        if strict or (want_i8 and not is_bf16 and i8_strict):
            # `int8=True` is strict for the int8 path: a load failure of an explicitly demanded int8
            # object raises rather than silently serving fp8.
            raise
        if want_i8 and not is_bf16:
            _log_once("int8 load failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("load failed: %s" % e)
        return None, "load_failed"
    if attn is None:
        if is_bf16:
            if d64:
                # Only reachable when `SK1_D64B_ARM` is set (otherwise the refusal above fired) and
                # the bf16 D=64 object is missing or will not load.
                if b_strict or strict:
                    raise RuntimeError("the bf16 head_dim-64 SK1 object %s is not wired (%s)"
                                       % (SK1_HSACO_NAME_D64B, _BACKEND_D64B_FAILED))
                _log_once("bf16 head_dim-64 object unavailable: %s" % _BACKEND_D64B_FAILED)
                return None, "no_d64b_object"
            name = SK1_HSACO_NAME_B if contiguous_hnd else SK1_HSACO_NAME_BN
            fail = _BACKEND_B_FAILED if contiguous_hnd else _BACKEND_BN_FAILED
            if b_strict or strict:
                raise RuntimeError("the bf16 SK1 object %s is not installed (%s)" % (name, fail))
            _log_once("bf16 object unavailable: %s" % fail)
            return None, "no_bf16_object"
        if want_i8:
            name = SK1_HSACO_NAME_I8 if contiguous_hnd else SK1_HSACO_NAME_I8N
            fail = _BACKEND_I8_FAILED if contiguous_hnd else _BACKEND_I8N_FAILED
            if i8_strict or strict:
                raise RuntimeError("the int8 SK1 object %s is not installed (%s)" % (name, fail))
            _log_once("int8 object unavailable (%s), falling back to the fp8 SK1 objects" % fail)
            return _i8_fallback()
        if d64:
            if strict:
                raise RuntimeError("the head_dim-64 SK1 object %s is not wired (%s)"
                                   % (SK1_HSACO_NAME_D64, _BACKEND_D64_FAILED))
            _log_once("head_dim-64 object unavailable: %s" % _BACKEND_D64_FAILED)
            return None, "no_d64_object"
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
        hp = _hip_prologue(_arch)   # `None` unless SAGEATTN_SK1_PROLOGUE=hip and it loaded
        if is_bf16:
            # The same fused K/V kernels, called with bf16 tensors. Every `tl.load` in
            # `quant_kv_v8t`/`quant_triton` is followed by `.to(tl.float32)` and a bf16 -> fp32
            # widening is exact for every bit pattern, so the fp8 bytes and the fp32 scales are what
            # the fp16 path produces for the same values. The contiguous path skips the discarded
            # `quant_rows_fp8(q)`; the strided path reuses the K/V-only prologue as is. `out` is bf16,
            # the kernel's own O type.
            if contiguous_hnd:
                qr = q.reshape(BH, N, D).contiguous()
                # The HIP prologue first; the Triton call is the fallback.
                res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_b as QB
                    res = QB.prologue_bf16_fused(k, v, n_pad=n_pad, smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                out = torch.empty(BH, N, D, device=q.device, dtype=q.dtype)
            else:
                res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                       if hp is not None else None)
                if res is None:
                    from . import quant_kv_v8t_n as QN
                    res = QN.prologue_fp8_fused_strided(kk, vv, n_pad=n_pad,
                                                        smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                if tensor_layout == "NHD":
                    out = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
                else:
                    out = torch.empty(BH, N, D, device=q.device, dtype=q.dtype)
        elif want_i8:
            # K -> int8 (scale amax/127, round-to-nearest-even, clamp), V -> fp8 e4m3. Same pad
            # contract, same `S` staging, same launch below. The strided module also serves
            # contiguous HND (it is the 18-argument form) but the contiguous-HND cell keeps the plain
            # 12-argument object, exactly as the fp8 pair does.
            from . import quant_kv_v8t_i8_n as QI8N
            if contiguous_hnd:
                qr = q.reshape(BH, N, D).contiguous()
                res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=True)
                       if hp is not None else None)
                if res is None:
                    res = QI8N.prologue_i8_fused_contig(k, v, n_pad=n_pad,
                                                        smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
            else:
                res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=True)
                       if hp is not None else None)
                if res is None:
                    res = QI8N.prologue_i8_fused_strided(kk, vv, n_pad=n_pad,
                                                         smooth_k=bool(smooth_k))
                k8, sk, v8t, sv = res
                if tensor_layout == "NHD":
                    out = torch.empty(B, N, H, D, device=q.device, dtype=torch.float16)
                else:
                    out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        elif contiguous_hnd and not d64:
            qr = q.reshape(BH, N, D).contiguous()
            res = (hp.quant_kv(k, v, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                   if hp is not None else None)
            if res is None:
                _, _, k8, sk, v8t, sv = QV.prologue_fp8_fused(q, k, v, n_pad=n_pad,
                                                              smooth_k=bool(smooth_k))
            else:
                k8, sk, v8t, sv = res
            out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        else:
            # The widened prologue reads K/V straight out of the strided view (no copy) and skips
            # quantising Q, whose int8 result and scales the SK1 kernel would discard anyway.
            res = (hp.quant_kv(kk, vv, n_pad=n_pad, smooth_k=bool(smooth_k), int8=False)
                   if hp is not None else None)
            if res is None:
                from . import quant_kv_v8t_n as QN
                res = QN.prologue_fp8_fused_strided(kk, vv, n_pad=n_pad,
                                                    smooth_k=bool(smooth_k))
            k8, sk, v8t, sv = res
            if tensor_layout == "NHD":
                out = torch.empty(B, N, H, D, device=q.device, dtype=torch.float16)
            else:
                out = torch.empty(BH, N, D, device=q.device, dtype=torch.float16)
        # Only the 12-argument objects want `S`. When they do, this stages `SV' = SV/S` and returns S
        # for the kernel's epilogue. For every input with |V| <= 448, S == 1.0 and `sv` comes out
        # bit-for-bit unchanged.
        sv_s = fold_v_scale(sv) if attn.with_s else None
    except Exception as e:                                          # noqa: BLE001
        if strict or (want_i8 and not is_bf16 and i8_strict):
            raise
        if want_i8 and not is_bf16:
            # A broken int8 prologue must not cost the call its fp8 SK1 service.
            _log_once("int8 prologue failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("prologue failed: %s" % e)
        return None, "prologue_failed"
    try:
        if contiguous_hnd and not d64:
            attn.launch(qr, k8, sk, v8t, sv, None, out, H, N, n_pad, sm, bool(is_causal), s=sv_s)
        else:
            # the widened kernel is addressed in the logical (B, H, N, D) reading of the caller's
            # tensor: for NHD that is the transpose view; `out` is written with the same pattern.
            # The D=64 object is the 18-argument strided form even for contiguous HND, so it always
            # takes this branch.
            attn.launch_strided(qq, k8, sk, v8t, sv, None,
                                out.view(B, H, N, D) if tensor_layout == "HND"
                                else out.transpose(1, 2),
                                H, N, n_pad, sm, bool(is_causal), s=sv_s)
    except Exception as e:
        if strict or (want_i8 and not is_bf16 and i8_strict):
            raise
        if want_i8 and not is_bf16:
            # Likewise for a failed int8 launch.
            _log_once("int8 launch failed, falling back to the fp8 SK1 objects: %s" % e)
            return _i8_fallback()
        _log_once("launch failed: %s" % e)
        return None, "launch_failed"
    if tensor_layout == "NHD":
        return out, None
    return out.view(B, H, N, D), None
