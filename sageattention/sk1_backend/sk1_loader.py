#!/usr/bin/env python3
"""Load and launch a raw `.hsaco` from Python with ctypes, on torch's own HIP runtime.

A code object has no C++ imports, so the `c10` symbol ABI mismatch that breaks a `.pyd` extension
built against a different torch cannot arise. Everything here is a plain C-ABI call with integer
and pointer arguments.

Three things this file is careful about:

1. The same HIP runtime DLL torch uses. `hipModuleLoadData` must be called on the loaded
   `amdhip64_7.dll`, not on a second copy, or device pointers and streams are not interchangeable.
   In the ROCm SDK Python packages, `_rocm_sdk_core\\bin\\amdhip64_7.dll` (the one torch loads) and
   `_rocm_sdk_devel\\bin\\amdhip64_7.dll` were observed to be two different builds, and an unguarded
   `ctypes.WinDLL` on the devel path puts a second HIP runtime in the process. `rocm_bin_dir()`
   therefore resolves to the image that is already loaded, and `HipRuntime.same_as_torch` records
   the enumeration as evidence.
2. Torch's current stream. `hipModuleLaunchKernel` is given `torch.cuda.current_stream().cuda_stream`,
   a Python-level integer handle, so no C++ object crosses the boundary and the kernel is ordered
   against the torch ops that produced K8/SK/V8T/SV.
3. Argument lifetime. `kernelParams` holds pointers to the argument values and HIP reads them
   asynchronously, so the values must outlive the launch. `launch()` returns them and
   `Sk1Attn.__call__` keeps them on the object until the next call.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

HIP_SUCCESS = 0

# hipModuleLaunchKernel(hipFunction_t, gx, gy, gz, bx, by, bz, sharedMemBytes, stream,
#                       kernelParams, extra)  -- exactly 11 entries in declaration order.
# (Writing this as [c_void_p, c_uint]*6 would expand to 12 alternating entries and mis-type
#  kernelParams as c_uint.)
_LAUNCH_ARGTYPES = ([ctypes.c_void_p] + [ctypes.c_uint] * 6 +
                    [ctypes.c_uint, ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)])


def _enum_amdhip_modules() -> list:
    """Every loaded module whose base name starts with `amdhip` -- full path + HMODULE."""
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    proc = k32.GetCurrentProcess()
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                         ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                           wintypes.LPWSTR, ctypes.c_uint]
    arr = (wintypes.HMODULE * 2048)()
    need = ctypes.c_uint()
    if not psapi.EnumProcessModules(proc, arr, ctypes.sizeof(arr), ctypes.byref(need)):
        return []
    out = []
    buf = ctypes.create_unicode_buffer(4096)
    for i in range(need.value // ctypes.sizeof(wintypes.HMODULE)):
        psapi.GetModuleFileNameExW(proc, arr[i], buf, 4096)
        p = buf.value
        if os.path.basename(p).lower().startswith("amdhip"):
            out.append(dict(path=p, hmodule=ctypes.cast(arr[i], ctypes.c_void_p).value))
    return out


def _candidate_bins() -> list:
    cands = []
    d = os.environ.get("SK1_ROCM_BIN")
    if d:
        cands.append(d)
    import site
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        # `_rocm_sdk_core` first: that is the copy torch's own wheel loads. The `_rocm_sdk_devel`
        # `amdhip64_7.dll` has been a different build, and loading it puts a second HIP runtime in
        # the process.
        cands.append(os.path.join(sp, "_rocm_sdk_core", "bin"))
        cands.append(os.path.join(sp, "_rocm_sdk_devel", "bin"))
    return cands


def rocm_bin_dir() -> str:
    """Directory holding the `amdhip64_7.dll` that torch already loaded, else the best fallback.

    Device pointers and the stream must be shared with torch, which means the same runtime image.
    Windows' loader keys on the resolved path, so loading the DLL from a different directory yields
    a second, independent HIP runtime in the same process. `_rocm_sdk_core\\bin` (torch's) and
    `_rocm_sdk_devel\\bin` have been seen to hold two different builds, and both ended up loaded,
    which is why this function prefers the already-loaded image.
    """
    override = os.environ.get("SK1_HIP_DLL")
    if override and os.path.isfile(override):
        return os.path.dirname(override)
    # make sure torch has had its chance to load its own copy before we enumerate
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.current_stream()
    except Exception:
        pass
    for m in _enum_amdhip_modules():
        return os.path.dirname(m["path"])
    for c in _candidate_bins():
        if c and os.path.isfile(os.path.join(c, "amdhip64_7.dll")):
            return c
    raise RuntimeError("amdhip64_7.dll not found; set SK1_ROCM_BIN or SK1_HIP_DLL")


class HipRuntime:
    """Thin ctypes binding to the HIP runtime that is already loaded in this process."""

    DLL_NAME = "amdhip64_7.dll"

    def __init__(self, bin_dir: str | None = None):
        self.bin_dir = bin_dir or rocm_bin_dir()
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(self.bin_dir)   # amdhip64_7.dll has siblings on its search path
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self._k32.GetModuleHandleW.restype = wintypes.HMODULE
        self.dll_path = os.path.join(self.bin_dir, self.DLL_NAME)
        self.hip = ctypes.WinDLL(self.dll_path)
        self._bind()
        self.same_as_torch = self._verify_same_module()
        self.modules: list = []          # keep modules (and their images) alive
        self._keepalive = None           # the last launch's argument storage

    # ---------------------------------------------------------------- binding
    def _bind(self):
        h = self.hip
        h.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        h.hipModuleLoadData.restype = ctypes.c_int
        h.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        h.hipModuleGetFunction.restype = ctypes.c_int
        h.hipModuleLaunchKernel.argtypes = _LAUNCH_ARGTYPES
        h.hipModuleLaunchKernel.restype = ctypes.c_int
        h.hipModuleUnload.argtypes = [ctypes.c_void_p]
        h.hipModuleUnload.restype = ctypes.c_int
        for fn in ("hipGetErrorString", "hipGetErrorName"):
            f = getattr(h, fn)
            f.argtypes = [ctypes.c_int]
            f.restype = ctypes.c_char_p
        h.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        h.hipStreamSynchronize.restype = ctypes.c_int

    def err(self, rc: int) -> str:
        try:
            name = self.hip.hipGetErrorName(rc).decode()
            msg = self.hip.hipGetErrorString(rc).decode()
            return "%d (%s: %s)" % (rc, name, msg)
        except Exception:
            return str(rc)

    def loaded_amdhip_modules(self) -> list:
        return _enum_amdhip_modules()

    def _verify_same_module(self) -> dict:
        """Evidence that ctypes and torch share one HIP runtime image.

        `WinDLL` and `GetModuleHandleW` both key on the module *name*, so equality of the handles
        means Windows handed back the already-loaded image rather than mapping a second one.  The
        torch side is recorded as the stream handle's non-emptiness plus the DLL's own path.
        """
        got = ctypes.cast(self.hip._handle, ctypes.c_void_p).value
        want = self._k32.GetModuleHandleW(self.DLL_NAME)
        mods = _enum_amdhip_modules()
        rec = dict(dll=self.DLL_NAME, bin_dir=self.bin_dir, dll_path=self.dll_path,
                   ctypes_hmodule=got, getmodulehandle_hmodule=want,
                   handle_equal=bool(got and want and got == want),
                   loaded_amdhip_modules=mods, n_amdhip_images=len(mods))
        try:
            import torch
            rec["torch_version"] = torch.__version__
            rec["torch_hip_version"] = getattr(torch.version, "hip", None)
            rec["torch_cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                rec["stream_handle"] = int(torch.cuda.current_stream().cuda_stream)
        except Exception as e:                                    # pragma: no cover
            rec["torch_error"] = repr(e)
        return rec

    # ---------------------------------------------------------------- module / function
    def load_module(self, hsaco):
        """`hsaco`: path or raw bytes of an `elf64-amdgpu` code object (NOT a `__CL` container)."""
        image = open(hsaco, "rb").read() if isinstance(hsaco, (str, os.PathLike)) else bytes(hsaco)
        if image[:4] != b"\x7fELF":
            raise RuntimeError("hipModuleLoadData wants a raw ELF; got magic %r "
                               "(unbundle the clang-offload-bundler container first)" % image[:4])
        buf = ctypes.create_string_buffer(image, len(image))       # HIP does not copy
        mod = ctypes.c_void_p()
        rc = self.hip.hipModuleLoadData(ctypes.byref(mod), buf)
        if rc:
            raise RuntimeError("hipModuleLoadData rc=%s" % self.err(rc))
        self.modules.append((mod, buf, image))                     # keep alive
        return mod

    def get_function(self, mod, symbol: str):
        fn = ctypes.c_void_p()
        rc = self.hip.hipModuleGetFunction(ctypes.byref(fn), mod, symbol.encode())
        if rc:
            raise RuntimeError("hipModuleGetFunction(%r) rc=%s" % (symbol, self.err(rc)))
        return fn

    # ---------------------------------------------------------------- launch
    def launch(self, fn, grid, block, shared_bytes, stream, args):
        """`args`: list of ctypes values in the KERNEL's declaration order.

        HIP dereferences `kernelParams` asynchronously, so BOTH the pointer array and the argument
        values themselves must outlive the launch.  The runtime keeps them alive internally
        (`self._keepalive`) as well as returning them, because a caller that ignores the return
        value would silently free the argument storage.
        """
        vals = [ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args]
        arr = (ctypes.c_void_p * len(vals))(*vals)
        rc = self.hip.hipModuleLaunchKernel(fn, *[ctypes.c_uint(x) for x in grid],
                                            *[ctypes.c_uint(x) for x in block],
                                            ctypes.c_uint(shared_bytes),
                                            ctypes.c_void_p(int(stream)), arr, None)
        if rc:
            raise RuntimeError("hipModuleLaunchKernel rc=%s" % self.err(rc))
        self._keepalive = (args, vals, arr)
        return vals, arr

    def sync(self, stream=0):
        rc = self.hip.hipStreamSynchronize(ctypes.c_void_p(int(stream)))
        if rc:
            raise RuntimeError("hipStreamSynchronize rc=%s" % self.err(rc))


def torch_stream():
    import torch
    return int(torch.cuda.current_stream().cuda_stream)


def _strided_triple(t, name):
    """`(row, head, batch)` element strides of a logical `(B, H, N, D)` tensor.

    The widened kernel takes the addressing as three integers instead of assuming the contiguous
    `(B*H, N, 128)` form. Only two patterns are legal, and both are views of a contiguous storage, so
    the address map is injective by construction and cannot alias:

    * `t.is_contiguous()` -- contiguous HND, `(rs, hs, bs) = (D, N*D, H*N*D)`;
    * `t.transpose(1, 2).is_contiguous()` -- the physical NHD layout read as HND, which is exactly
      what `rearrange(q, "B L (H D) -> B H L D")` produces and what NHD `(B, N, H, D)` input
      transposes to, `(rs, hs, bs) = (H*D, D, N*H*D)`.

    Anything else is refused: an arbitrary stride set could overlap itself and the launch is raw.
    """
    if t.dim() != 4:
        raise ValueError("%s must be 4-D (B,H,N,D); got rank %d" % (name, t.dim()))
    if t.is_contiguous():
        return int(t.stride(2)), int(t.stride(1)), int(t.stride(0)), "hnd_contig"
    if t.transpose(1, 2).is_contiguous():
        return int(t.stride(2)), int(t.stride(1)), int(t.stride(0)), "nhd_phys"
    raise ValueError("%s is neither contiguous HND nor a physically-NHD (B,H,N,D) view: shape=%s "
                     "strides=%s" % (name, tuple(t.shape), tuple(t.stride())))


class Sk1Attn:
    """SK1's two entry points behind a torch-tensor interface.

    Tensor contract (exactly the kernel's):  Q (B*H, N, 128) fp16 RAW; K8 (B*H, N_pad, 128) e4m3
    zero-padded; SK (B*H, N_pad) fp32; V8T (B*H, 128, N_pad) e4m3 zero-padded (V TRANSPOSED);
    SV (B*H, N_pad) fp32; VMEAN (B*H, 128) fp32 or None; O (B*H, N, 128) fp16.
    `H` is the head count used to fold (b, h) out of `bh`; N_pad must be a multiple of 64.

    The `sk1_t4a1s` object takes a 12th argument, `S` (B*H,) fp32: the per-`(b,h)` rescale
    `S = clamp_min(max_j sv(j), 1)` that the prologue applied to `SV`. It is inserted between `VMEAN`
    and `O`, so the kernel argument order is `Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale`.
    `with_s` selects it; it defaults to `True` for the `sk1_t4a1s` symbol pair and `False` otherwise,
    so the 11-argument `sk1_t4a1` keeps working.
    """

    #: The default symbols are the `sk1_t4a1` pair. `sk1_t1`'s pair is
    #: `("sk1t1_attn_fwd_c0", "sk1t1_attn_fwd_c1")`; the `sk1_t4a1s` pair is
    #: `("sk1t4a1s_attn_fwd_c0", "sk1t4a1s_attn_fwd_c1")` and takes `S`.
    DEFAULT_SYMBOLS = ("sk1t4a1_attn_fwd_c0", "sk1t4a1_attn_fwd_c1")

    #: The argument order of the `sk1_t4a1s` kernel, checked against its emitted ISA:
    #: `s_load_b512 s[0:15], s[0:1], 0x0` covers Q..O at offsets 0x00..0x3F and
    #: `s_load_b128 s[16:19], s[0:1], 0x40` covers H, N, N_pad and sm.
    ARG_ORDER_WITH_S = ("Q", "K8", "SK", "V8T", "SV", "VMEAN", "S", "O", "H", "N", "N_pad", "sm")

    #: The widened object `sk1_t4a1n` takes six extra integers after `sm_scale`: the
    #: `(row, head, batch)` element strides of `Q` and of `O` (see `kernels/hip/sk1_t4a1n.hip`).
    #: With `(q_rs, q_hs, q_bs) = (128, N*128, H*N*128)` and the same for `O`, the kernel's addresses
    #: are identical to `sk1_t4a1s`'s. Contiguous HND is still routed to `sk1_t4a1s` (routing is in
    #: `__init__.py`), so that path keeps its original code object.
    ARG_ORDER_STRIDED = ("Q", "K8", "SK", "V8T", "SV", "VMEAN", "S", "O", "H", "N", "N_pad", "sm",
                         "q_rs", "q_hs", "q_bs", "o_rs", "o_hs", "o_bs")

    def __init__(self, hsaco_path: str, rt: HipRuntime | None = None, symbols=None, with_s=None,
                 strided=None):
        """`symbols` = (non-causal, causal) entry points.  The packaged default is the `sk1_t4a1`
        variant's pair (`sk1t4a1_attn_fwd_c0/_c1`); the `_check` contract guard runs before every
        launch.  `with_s` selects the 12-argument form (see the class doc)."""
        self.rt = rt or HipRuntime()
        self.hsaco_path = hsaco_path
        self.symbols = tuple(symbols) if symbols else self.DEFAULT_SYMBOLS
        if len(self.symbols) != 2:
            raise ValueError("symbols must be (non-causal, causal); got %r" % (self.symbols,))
        # `with_s` defaults from the symbol pair, so a caller that only changes `symbols` cannot
        # silently launch the 12-argument kernel with 11 arguments (a GPU page fault).
        self.with_s = (any("t4a1s" in s for s in self.symbols) if with_s is None else bool(with_s))
        # Same guard for the six-stride form: defaulting from the symbol pair means a caller that
        # only swaps `symbols` cannot launch the 18-argument kernel with 12 arguments.
        self.strided = (any("t4a1n" in s for s in self.symbols) if strided is None
                        else bool(strided))
        self.module = self.rt.load_module(hsaco_path)
        self.funcs = {sym: self.rt.get_function(self.module, sym) for sym in self.symbols}
        self._keep = None

    @staticmethod
    def _check(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=None, with_s=False):
        # The launch is raw: nothing downstream bounds-checks, and an undersized buffer is a GPU page
        # fault (Windows shows an AMD bug-report popup). Refuse any tensor that does not match the
        # contract exactly, before it reaches the device.
        import torch
        # This must come first. A null `S` on the 12-argument object is dereferenced by the kernel's
        # `Sb = S[bh]` load, so it is refused before any other tensor is inspected.
        if with_s and s is None:
            raise ValueError("this SK1 object takes the per-head V scale `S` argument; s=None was passed")
        N, N_pad, H = int(N), int(N_pad), int(H)
        if N < 1 or N_pad % 64 or N_pad < N:
            # `% 64` must be escaped: unescaped, Python parses it as a format spec and raises
            # `ValueError: unsupported format character` instead of this message.
            raise ValueError("need N >= 1, N_pad %% 64 == 0 and N_pad >= N (N=%d, N_pad=%d)"
                             % (N, N_pad))
        bh = q.shape[0]
        want = {
            "q": (q, (bh, N, 128), torch.float16),
            "k8": (k8, (bh, N_pad, 128), None),
            "sk": (sk, (bh, N_pad), torch.float32),
            "v8t": (v8t, (bh, 128, N_pad), None),
            "sv": (sv, (bh, N_pad), torch.float32),
            "out": (out, (bh, N, 128), torch.float16),
        }
        if vmean is not None:
            want["vmean"] = (vmean, (bh, 128), torch.float32)
        # The 12th kernel argument (already required to be non-None at the top).
        if with_s:
            want["s"] = (s, (bh,), torch.float32)
        for name, (t, shape, dtype) in want.items():
            if tuple(t.shape) != shape:
                raise ValueError("%s shape %s != %s" % (name, tuple(t.shape), shape))
            if dtype is not None and t.dtype != dtype:
                raise ValueError("%s dtype %s != %s" % (name, t.dtype, dtype))
            if dtype is None and t.element_size() != 1:
                raise ValueError("%s must be a 1-byte e4m3 tensor, got %s" % (name, t.dtype))
            if not t.is_cuda or not t.is_contiguous():
                raise ValueError("%s must be a contiguous device tensor" % name)
            if t.device != q.device:
                raise ValueError("%s is on %s, q is on %s" % (name, t.device, q.device))

    @staticmethod
    def _check_strided(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=None):
        """`_check` for the 18-argument object.

        Same contract as `_check`, except that `q` and `out` are logical `(B, H, N, D)` tensors that
        may be strided.  Their `(row, head, batch)` strides must be one of the two legal patterns
        (`_strided_triple`); `K8`/`SK`/`V8T`/`SV`/`VMEAN`/`S` must still be contiguous.  Returns
        `(q_triple, o_triple)` for `launch_strided`.
        """
        import torch
        if s is None:
            raise ValueError("this SK1 object takes the per-head V scale `S` argument; s=None was passed")
        N, N_pad, H = int(N), int(N_pad), int(H)
        if N < 1 or N_pad % 64 or N_pad < N:
            raise ValueError("need N >= 1, N_pad %% 64 == 0 and N_pad >= N (N=%d, N_pad=%d)"
                             % (N, N_pad))
        if q.dim() != 4 or out.dim() != 4:
            raise ValueError("q/out must be 4-D (B,H,N,D); got ranks %d/%d" % (q.dim(), out.dim()))
        B, Hq, Nq, Dq = (int(x) for x in q.shape)
        if Hq != H:
            raise ValueError("q.shape[1]=%d != H=%d" % (Hq, H))
        if (Nq, Dq) != (N, 128):
            raise ValueError("q is %s, expected (B,%d,%d,128)" % (tuple(q.shape), H, N))
        bh = B * H
        qtr = _strided_triple(q, "q")
        otr = _strided_triple(out, "out")
        # The Q load is a 16-byte `f16x8` (see `sk1_t4a1n.hip`), so every (b,h,row) start must be
        # 16-byte aligned: an 8-element multiple on each stride plus an aligned base pointer.
        for nm, tr in (("q", qtr), ("out", otr)):
            rs, hs, bs = tr[0], tr[1], tr[2]
            if rs <= 0 or hs <= 0 or bs <= 0:
                raise ValueError("%s strides must be positive; got %s" % (nm, (rs, hs, bs)))
            if rs % 8 or hs % 8 or bs % 8:
                raise ValueError("%s strides must be multiples of 8 elements (16-byte Q loads); "
                                 "got %s" % (nm, (rs, hs, bs)))
        if q.data_ptr() % 16:
            raise ValueError("q.data_ptr()=0x%x is not 16-byte aligned" % q.data_ptr())
        want = (
            ("q", q, (B, H, N, 128), torch.float16, False),
            ("out", out, (B, H, N, 128), torch.float16, False),
            ("k8", k8, (bh, N_pad, 128), None, True),
            ("sk", sk, (bh, N_pad), torch.float32, True),
            ("v8t", v8t, (bh, 128, N_pad), None, True),
            ("sv", sv, (bh, N_pad), torch.float32, True),
        )
        for name, t, shape, dtype, need_contig in want:
            if tuple(t.shape) != shape:
                raise ValueError("%s shape %s != %s" % (name, tuple(t.shape), shape))
            if dtype is not None and t.dtype != dtype:
                raise ValueError("%s dtype %s != %s" % (name, t.dtype, dtype))
            if dtype is None and t.element_size() != 1:
                raise ValueError("%s must be a 1-byte e4m3 tensor, got %s" % (name, t.dtype))
            if not t.is_cuda:
                raise ValueError("%s must be a device tensor" % name)
            if t.device != q.device:
                raise ValueError("%s is on %s, q is on %s" % (name, t.device, q.device))
            if need_contig and not t.is_contiguous():
                raise ValueError("%s must be a contiguous device tensor" % name)
        if vmean is not None:
            if tuple(vmean.shape) != (bh, 128) or vmean.dtype != torch.float32:
                raise ValueError("vmean must be (%d,128) fp32; got %s %s"
                                 % (bh, tuple(vmean.shape), vmean.dtype))
            if not vmean.is_cuda or not vmean.is_contiguous() or vmean.device != q.device:
                raise ValueError("vmean must be a contiguous device tensor on q's device")
        if tuple(s.shape) != (bh,) or s.dtype != torch.float32:
            raise ValueError("s must be (%d,) fp32; got %s %s" % (bh, tuple(s.shape), s.dtype))
        if not s.is_cuda or not s.is_contiguous() or s.device != q.device:
            raise ValueError("s must be a contiguous device tensor on q's device")
        return qtr, otr

    def launch(self, q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, sm_scale, causal,
               stream=None, s=None):
        self._check(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=s, with_s=self.with_s)
        c = ctypes
        args = [c.c_void_p(q.data_ptr()), c.c_void_p(k8.data_ptr()), c.c_void_p(sk.data_ptr()),
                c.c_void_p(v8t.data_ptr()), c.c_void_p(sv.data_ptr()),
                c.c_void_p(vmean.data_ptr() if vmean is not None else 0)]
        if self.with_s:
            # `S` sits between `VMEAN` and `O` (ARG_ORDER_WITH_S), matching the kernel's kernarg
            # layout.
            args.append(c.c_void_p(s.data_ptr()))
        args += [c.c_void_p(out.data_ptr()),
                 c.c_int(int(H)), c.c_int(int(N)), c.c_int(int(N_pad)), c.c_float(float(sm_scale))]
        bh = q.shape[0]
        B = int(bh) // int(H)
        if B * int(H) != int(bh):
            raise ValueError("q.shape[0]=%d is not H=%d times an integer batch size" % (bh, H))
        # grid = (ceil(N/128), H, B); the kernel folds bh = blockIdx.z*H + blockIdx.y
        grid = ((N + 127) // 128, int(H), B)
        st = torch_stream() if stream is None else int(stream)
        self._keep = self.rt.launch(self.funcs[self.symbols[1] if causal else self.symbols[0]],
                                    grid, (256, 1, 1), 0, st, args)
        return out

    def launch_strided(self, q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, sm_scale, causal,
                       stream=None, s=None):
        """Launch the 18-argument widened kernel on logical `(B, H, N, D)` `q`/`out`.

        `q`/`out` may be contiguous HND or physically-NHD views (`_strided_triple`); every other
        tensor is the same as `launch`.  `out` is written in place and returned.
        """
        if not self.strided:
            raise ValueError("this SK1 object is not the strided form (symbols=%r)"
                             % (self.symbols,))
        qtr, otr = self._check_strided(q, k8, sk, v8t, sv, vmean, out, H, N, N_pad, s=s)
        c = ctypes
        args = [c.c_void_p(q.data_ptr()), c.c_void_p(k8.data_ptr()), c.c_void_p(sk.data_ptr()),
                c.c_void_p(v8t.data_ptr()), c.c_void_p(sv.data_ptr()),
                c.c_void_p(vmean.data_ptr() if vmean is not None else 0)]
        if self.with_s:
            args.append(c.c_void_p(s.data_ptr()))
        args += [c.c_void_p(out.data_ptr()),
                 c.c_int(int(H)), c.c_int(int(N)), c.c_int(int(N_pad)), c.c_float(float(sm_scale)),
                 c.c_int(int(qtr[0])), c.c_int(int(qtr[1])), c.c_int(int(qtr[2])),
                 c.c_int(int(otr[0])), c.c_int(int(otr[1])), c.c_int(int(otr[2]))]
        B = int(q.shape[0])
        # grid = (ceil(N/128), H, B); the kernel folds bh = blockIdx.z*H + blockIdx.y
        grid = ((N + 127) // 128, int(H), B)
        st = torch_stream() if stream is None else int(stream)
        self._keep = self.rt.launch(self.funcs[self.symbols[1] if causal else self.symbols[0]],
                                    grid, (256, 1, 1), 0, st, args)
        return out


if __name__ == "__main__":
    import json
    rt = HipRuntime()
    print(json.dumps(rt.same_as_torch, indent=1, sort_keys=True))

