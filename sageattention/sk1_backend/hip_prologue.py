"""The HIP KV prologue: a ctypes-launched code object that replaces the Triton prologue on the
default SK1 paths. Source: `kernels/hip/h34_prologue.hip`.

A second code object, `h36_prologue_d64.gfx1201.hsaco`, covers `head_dim == 64`. It is loaded
tolerantly and is not part of this package: when it is missing or will not load, D=64 calls refuse
(`no_symbol_d64`) and the caller runs the Triton D=64 prologue, while the D=128 object is unaffected.
D=64 is fp8 Q.K only. `int8=True` at D=64 refuses `int8_d64`, because the int8 Q.K objects are the
D=128-only `sk1_t6i` / `sk1_t6in` and no int8 D=64 attention object exists.

It produces exactly the four tensors the SK1 attention objects consume, and nothing else:

    k8  (BH, n_pad, D)   int8 (int8 arm) or e4m3 (fp8 arm), contiguous
    sk  (BH, n_pad)      fp32
    v8t (BH, D, n_pad)   e4m3, contiguous, transposed
    sv  (BH, n_pad)      fp32

`vmean` is not produced (the shipped path passes `None`) and `S` is still made by
`fold_v_scale(sv)` in torch. No Q is read: the fp8-contiguous Triton prologue's `quant_rows_fp8(q)`
result is discarded.

Two launches at most (`h34_kseq_mean_*` only when `smooth_k`, then `h34_quant_kv_*`), against 1-4
Triton launches otherwise. All addressing inside the kernels is 64-bit, so the prologue itself has no
2^31-element offset limit; `strided_view_reason` is still the acceptance rule, so the prologue serves
exactly the views the rest of SK1 accepts.

Loading and launching go through `sk1_loader.HipRuntime`, the same ctypes path the attention objects
use, so there is no second runtime, and the launch always carries a contract check.
"""
from __future__ import annotations

import collections
import ctypes
import os

import torch

from .sk1_loader import HipRuntime, strided_view_reason, torch_stream

HERE = os.path.dirname(os.path.abspath(__file__))
HSACO_NAME = "h34_prologue.gfx1201.hsaco"
HSACO_PATH = os.path.join(HERE, HSACO_NAME)


def hsaco_path_for(arch=None, name=None):
    """The D=128 prologue object for `arch` (`""` / `None` / `gfx1201` -> the gfx1201 object).

    The prologue is a code object like the attention objects, so it is per-arch:
    `h34_prologue.gfx1201.hsaco` / `h34_prologue_1200.hsaco`. The symbols are identical (same
    source, same commands, only `--offload-arch` differs), so only the path changes.
    """
    base = name or HSACO_NAME
    if arch in (None, "", "gfx1201"):
        return os.path.join(HERE, base)
    return os.path.join(HERE, base.replace(".gfx1201.hsaco", "_%s.hsaco" % arch.replace("gfx", "")))

#: The D=64 object is a separate code object, so the D=128 object and its sha256 are unaffected by
#: it. It has no int8 K symbol on purpose: the int8 Q.K path is the D=128-only `sk1_t6i` /
#: `sk1_t6in` pair, so every D=64 call that reaches a prologue is fp8 Q.K (`check` refuses
#: `int8=True` with `int8_d64`).
HSACO_NAME_D64 = "h36_prologue_d64.gfx1201.hsaco"
HSACO_PATH_D64 = os.path.join(HERE, HSACO_NAME_D64)

HD = 128          # the D=128 prologue
HD_D64 = 64       # the D=64 prologue
BN = 64           # rows per block, matches the Triton BLOCK_N=64
NT = 256          # threads per block
MEAN_BLOCK = (32, 8, 1)

#: (input dtype tag, K quantiser) -> symbol
SYMBOLS = {
    ("f16", "i8"): "h34_quant_kv_f16_i8",
    ("f16", "f8"): "h34_quant_kv_f16_f8",
    ("bf16", "f8"): "h34_quant_kv_bf16_f8",
}
MEAN_SYMBOLS = {"f16": "h34_kseq_mean_f16", "bf16": "h34_kseq_mean_bf16"}
#: D=64 has no int8 K variant (there is no int8 D=64 attention object to feed)
SYMBOLS_D64 = {
    ("f16", "f8"): "h36_quant_kv_f16_f8",
    ("bf16", "f8"): "h36_quant_kv_bf16_f8",
}
MEAN_SYMBOLS_D64 = {"f16": "h36_kseq_mean_f16", "bf16": "h36_kseq_mean_bf16"}

#: every reason `check` can return, so a test can enumerate them
REFUSALS = ("not_cuda", "rank", "shape", "dtype_mismatch", "bf16_int8", "dtype", "head_dim",
            "device", "n_pad", "n_lt_1", "no_symbol", "no_symbol_d64", "int8_d64", "launch",
            "mean_launch", "mean_dtype", "mean_head_dim", "mean_no_symbol_d64") + tuple(
    "k_" + r for r in ("rank", "stride_last", "stride_nonpos", "stride_align", "ptr_align",
                       "stride_overlap", "stride_int32")) + tuple(
    "v_" + r for r in ("rank", "stride_last", "stride_nonpos", "stride_align", "ptr_align",
                       "stride_overlap", "stride_int32"))


def n_pad_for(N: int) -> int:
    """SK1's N_pad contract: a multiple of 64."""
    return ((int(N) + BN - 1) // BN) * BN


class HipPrologue:
    """The loaded HIP prologue. One instance per process; `quant_kv` returns `None` (never raises)
    when the call is outside the contract or when an allocation or launch fails, so the caller can
    run the Triton prologue."""

    def __init__(self, hsaco_path=None, rt=None, hsaco_path_d64=None):
        self.path = hsaco_path or HSACO_PATH
        if not os.path.isfile(self.path):
            raise FileNotFoundError("the HIP prologue object is not built: %s "
                                    "(run tools/build_hsaco.py)" % self.path)
        self.rt = rt if rt is not None else HipRuntime()
        self.mod = self.rt.load_module(self.path)
        self.fns = {k: self.rt.get_function(self.mod, s) for k, s in SYMBOLS.items()}
        self.means = {k: self.rt.get_function(self.mod, s) for k, s in MEAN_SYMBOLS.items()}
        # The D=64 object is loaded tolerantly: a missing or unloadable one must not stop the D=128
        # prologue, so it degrades to `no_symbol_d64` and the caller runs the Triton D=64 prologue.
        self.path_d64 = hsaco_path_d64 or HSACO_PATH_D64
        self.fns64, self.means64, self.d64_error = {}, {}, None
        try:
            if not os.path.isfile(self.path_d64):
                raise FileNotFoundError("not built: %s" % self.path_d64)
            self.mod64 = self.rt.load_module(self.path_d64)
            self.fns64 = {k: self.rt.get_function(self.mod64, s) for k, s in SYMBOLS_D64.items()}
            self.means64 = {k: self.rt.get_function(self.mod64, s)
                            for k, s in MEAN_SYMBOLS_D64.items()}
        except Exception as e:  # noqa: BLE001 - D=64 degrades to the Triton prologue
            self.d64_error = "%s: %s" % (type(e).__name__, e)
        self.n_calls = 0
        self.n_refusals = 0
        self.n_mean_launches = 0
        self.n_kv_launches = 0
        self.refusal_reasons = collections.Counter()
        #: HIP dereferences `kernelParams` asynchronously, and this object issues two launches per
        #: call, so keep both argument blocks alive well past the launch (see `HipRuntime.launch`).
        self._keep = collections.deque(maxlen=128)

    def describe(self):
        d64 = ("%s sha256=%s symbols=%d" % (os.path.basename(self.path_d64),
                                            _sha256(self.path_d64)[:16],
                                            len(self.fns64) + len(self.means64))
               if not self.d64_error else "%s UNAVAILABLE (%s)" % (os.path.basename(self.path_d64),
                                                                   self.d64_error))
        return "%s sha256=%s symbols=%d | D=64: %s" % (
            os.path.basename(self.path), _sha256(self.path)[:16], len(self.fns) + len(self.means),
            d64)

    # ------------------------------------------------------------------ the contract
    def check(self, k, v, n_pad, int8):
        """`None` if this call is inside the prologue's contract, else a clean reason string.

        `head_dim` is `{64, 128}`. D=64 is fp8 Q.K only (`int8_d64`) because there is no int8 D=64
        attention object (see the module docstring).
        """
        if not (k.is_cuda and v.is_cuda):
            return "not_cuda"
        if k.dim() != 4 or v.dim() != 4:
            return "rank"
        if tuple(k.shape) != tuple(v.shape):
            return "shape"
        if k.dtype != v.dtype:
            return "dtype_mismatch"
        if k.dtype == torch.bfloat16:
            if int8:
                return "bf16_int8"
        elif k.dtype != torch.float16:
            return "dtype"
        D = int(k.shape[3])
        if D not in (HD, HD_D64):
            return "head_dim"
        if D == HD_D64 and int8:
            return "int8_d64"
        if k.device != v.device:
            return "device"
        r = strided_view_reason(k, "k")
        if r is not None:
            return "k_" + r
        r = strided_view_reason(v, "v")
        if r is not None:
            return "v_" + r
        N = int(k.shape[2])
        if N < 1:
            # N=0 would pass the n_pad checks below, then the mean kernel would compute 0/0 and
            # write NaN and the KV launch would get gridDim.x = 0. The served path already refuses
            # N<1, so this only guards the direct API.
            return "n_lt_1"
        if int(n_pad) < N or int(n_pad) % BN:
            return "n_pad"
        return None

    # ------------------------------------------------------------------ the launch
    def quant_kv(self, k, v, n_pad=None, smooth_k=False, int8=True, floor=1e-12):
        """`(k8, sk, v8t, sv)` for a logical `(B, H, N, D)` fp16/bf16 `k`/`v`, or `None` if refused.

        `k`/`v` may be strided views (any pattern `strided_view_reason` accepts); the four outputs
        are freshly allocated contiguous tensors, exactly as the Triton prologue returns them.

        `floor` is the int8-K scale floor only, as in the Triton int8 prologue. The fp8-K scale and
        V's scale use the hard-coded `1e-12` the Triton fp8 prologue uses and never `floor`.

        Never raises. A failed launch or an allocation failure is caught and returned as the refusal
        `"launch"`, so the caller runs the Triton prologue. Letting it propagate would surface as
        `prologue_failed`, which routes to the gfx12 native path instead of the Triton prologue.
        """
        if k.dim() != 4 or v.dim() != 4:
            return self._refuse("rank")
        B, H, N, D = (int(x) for x in k.shape)
        BH = B * H
        n_pad = n_pad_for(N) if n_pad is None else int(n_pad)
        reason = self.check(k, v, n_pad, int8)
        if reason is not None:
            return self._refuse(reason)
        dt = "bf16" if k.dtype == torch.bfloat16 else "f16"
        d64 = (D == HD_D64)
        fn = (self.fns64 if d64 else self.fns).get((dt, "i8" if int8 else "f8"))
        if fn is None:
            return self._refuse("no_symbol_d64" if d64 else "no_symbol")

        dev = k.device
        try:
            k8 = torch.empty((BH, n_pad, D),
                             dtype=(torch.int8 if int8 else torch.float8_e4m3fn), device=dev)
            sk = torch.empty((BH, n_pad), dtype=torch.float32, device=dev)
            v8t = torch.empty((BH, D, n_pad), dtype=torch.float8_e4m3fn, device=dev)
            sv = torch.empty((BH, n_pad), dtype=torch.float32, device=dev)

            mean = None
            if smooth_k:
                mean = torch.empty((BH, D), dtype=torch.float32, device=dev)
                self._launch_mean(k, mean, N, H)

            kargs = [ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
                     ctypes.c_void_p(k8.data_ptr()), ctypes.c_void_p(sk.data_ptr()),
                     ctypes.c_void_p(v8t.data_ptr()), ctypes.c_void_p(sv.data_ptr()),
                     ctypes.c_void_p(0 if mean is None else mean.data_ptr()),
                     ctypes.c_int64(N), ctypes.c_int64(n_pad), ctypes.c_int64(H),
                     ctypes.c_int64(k.stride(0)), ctypes.c_int64(k.stride(1)),
                     ctypes.c_int64(k.stride(2)),
                     ctypes.c_int64(v.stride(0)), ctypes.c_int64(v.stride(1)),
                     ctypes.c_int64(v.stride(2)),
                     ctypes.c_float(float(floor)), ctypes.c_int(1 if mean is not None else 0)]
            vals, arr = self.rt.launch(fn, (n_pad // BN, BH, 1), (NT, 1, 1), 0, torch_stream(), kargs)
        except Exception:  # noqa: BLE001 - the fallback contract is "never raises"
            return self._refuse("launch")
        self._keep.append((kargs, vals, arr))
        self.n_kv_launches += 1
        self.n_calls += 1
        return k8, sk, v8t, sv

    def _refuse(self, reason):
        self.n_refusals += 1
        self.refusal_reasons[reason] += 1
        return None

    # ------------------------------------------------------------------ the mean, on its own
    def _launch_mean(self, k, mean, N, H):
        dt = "bf16" if k.dtype == torch.bfloat16 else "f16"
        d64 = int(k.shape[3]) == HD_D64
        fn = (self.means64 if d64 else self.means).get(dt)
        if fn is None:
            raise RuntimeError("mean_no_symbol_d64" if d64 else "mean_no_symbol")
        margs = [ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(mean.data_ptr()),
                 ctypes.c_int64(N), ctypes.c_int64(H),
                 ctypes.c_int64(k.stride(0)), ctypes.c_int64(k.stride(1)),
                 ctypes.c_int64(k.stride(2))]
        vals, arr = self.rt.launch(fn, (int(mean.shape[0]), int(k.shape[3]) // 32, 1), MEAN_BLOCK,
                                   0, torch_stream(), margs)
        self._keep.append((margs, vals, arr))
        self.n_mean_launches += 1

    def kseq_mean(self, k):
        """The `smooth_k` sequence-axis mean `(B, H, D)` fp32, or `None` if refused.

        Exposed separately so a test can compare it against an fp64 reference without going through
        `quant_kv`. The summation is order-independent and deterministic: fp64 accumulation of
        exactly-widened fp16/bf16 inputs, one fp32 rounding at the end. That is "fp64-accumulated",
        not "exact" and not "correctly rounded": widening is exact, summing is not, and the `N*2^-53`
        bound is relative to the sum of `|x|`, so it says nothing when terms cancel.

        Never raises: an allocation or launch failure returns the refusal `"mean_launch"`.
        """
        if k.dim() != 4 or not k.is_cuda or k.dtype not in (torch.float16, torch.bfloat16):
            return self._refuse("mean_dtype")
        B, H, N, D = (int(x) for x in k.shape)
        if D not in (HD, HD_D64):
            return self._refuse("mean_head_dim")
        r = strided_view_reason(k, "k")
        if r is not None:
            return self._refuse("mean_" + r)
        if D == HD_D64 and not self.means64:
            return self._refuse("mean_no_symbol_d64")
        try:
            mean = torch.empty((B * H, D), dtype=torch.float32, device=k.device)
            self._launch_mean(k, mean, N, H)
        except Exception:  # noqa: BLE001 - never raises, extended to the mean entry
            return self._refuse("mean_launch")
        return mean.view(B, H, D)


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
