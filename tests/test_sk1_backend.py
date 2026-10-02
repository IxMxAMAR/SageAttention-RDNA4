"""Tests for the hand-written gfx1201 attention kernels, through the public `sageattn` API.

On gfx1201, head_dim 128 calls are served by the prebuilt kernels by default: fp16 by the int8 Q·K
kernel, bf16 by the bf16 fp8 kernel. These tests check accuracy against fp32 SDPA, edge lengths,
large V values, that NHD and HND inputs give the same bits, the environment switches and the
fallback. They are skipped when there is no gfx1201 GPU or no HIP runtime.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F


def _have_gfx1201():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return arch.split(":", 1)[0] == "gfx1201"


pytestmark = pytest.mark.skipif(not _have_gfx1201(), reason="needs a gfx1201 GPU and a HIP runtime")

# Relative RMS error of the quantized paths against fp32 SDPA on N(0,1) inputs is about 0.03 to 0.055.
MAX_REL_RMS = 0.08

HEAD_DIM = 128
DTYPES = [torch.float16, torch.bfloat16]


def _sageattn():
    from sageattention import sageattn
    return sageattn


def _rand_qkv(b, h, n, seed=0, layout="HND", dtype=torch.float16, head_dim=HEAD_DIM):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (b, h, n, head_dim) if layout == "HND" else (b, n, h, head_dim)
    return [torch.randn(shape, device="cuda", dtype=dtype, generator=g) for _ in range(3)]


def _rel_rms(out, ref):
    out = out.float()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _reference(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def _run_fresh(code, **env):
    # The SAGEATTN_* switches are read at import time, so these checks need a fresh interpreter.
    res = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ, **env),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return [l for l in res.stdout.splitlines() if l.startswith("RESULT ")][-1].split()[1:]


@pytest.mark.parametrize("dtype", DTYPES)
def test_hnd_matches_sdpa(dtype):
    q, k, v = _rand_qkv(1, 8, 512, dtype=dtype)
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert out.dtype == dtype and out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, False)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 15, 63, 64, 65, 127, 129, 300, 1000])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_edge_lengths(n, causal, smooth_k, dtype):
    q, k, v = _rand_qkv(1, 2, n, seed=n, dtype=dtype)
    out = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal, smooth_k=smooth_k)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, causal)) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_v_outlier_stays_finite(dtype):
    # All queries are identical and one key is aligned with them, so that key gets almost all of
    # the softmax weight. Its V row has |V| = 20000, far above the e4m3 maximum of 448. A kernel
    # that folds p * amax_v into the fp8 conversion without a clamp overflows here.
    n, outlier = 1024, 300
    q, k, v = _rand_qkv(1, 2, n, seed=1, dtype=dtype)
    q[:] = q[:, :, :1]
    k[:, :, outlier] = 4 * q[:, :, 0]
    v[:, :, outlier] = 20000.0
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert (~torch.isfinite(out)).sum().item() == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_nhd_equals_hnd_bitwise(causal, dtype):
    n = 300
    q, k, v = _rand_qkv(2, 4, n, seed=3, dtype=dtype)
    out_hnd = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal)
    qn, kn, vn = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out_nhd = _sageattn()(qn, kn, vn, tensor_layout="NHD", is_causal=causal)
    assert out_nhd.shape == (2, n, 4, HEAD_DIM)
    assert torch.equal(out_nhd.transpose(1, 2), out_hnd)


def test_int8_is_the_fp16_default():
    # The default fp16 path computes Q·K in int8, so it differs from the fp8 kernel and should be at
    # least as close to the fp32 reference.
    q, k, v = _rand_qkv(1, 8, 1024, seed=5)
    ref = _reference(q, k, v, False)
    out_int8 = _sageattn()(q, k, v, tensor_layout="HND")
    out_fp8 = _sageattn()(q, k, v, tensor_layout="HND", sk1_int8=False)
    assert not torch.equal(out_int8, out_fp8)
    assert _rel_rms(out_int8, ref) <= 1.05 * _rel_rms(out_fp8, ref)


def test_int8_switch_off_gives_the_fp8_kernel():
    code = """
        import torch
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16, generator=g)
                   for _ in range(3)]
        print("RESULT", torch.equal(sageattn(q, k, v), sageattn(q, k, v, sk1_int8=False)))
    """
    assert _run_fresh(code, SAGEATTN_SK1_INT8="0") == ["True"]


@pytest.mark.parametrize("env", [{"SAGEATTN_SK1_BACKEND": "0"}, {"SAGEATTN_SK1_BF16": "0"}])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_switched_off_still_works(env, dtype):
    code = """
        import torch
        import torch.nn.functional as F
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.%s, generator=g)
                   for _ in range(3)]
        out = sageattn(q, k, v, tensor_layout="HND")
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        rel = ((out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print("RESULT", bool(torch.isfinite(out).all()), rel)
    """ % dtype
    finite, rel = _run_fresh(code, **env)
    assert finite == "True"
    assert float(rel) < MAX_REL_RMS


@pytest.mark.parametrize("dtype", DTYPES)
def test_head_dim_64_falls_back(dtype):
    q, k, v = _rand_qkv(1, 4, 512, seed=7, dtype=dtype, head_dim=64)
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, False)) < MAX_REL_RMS
