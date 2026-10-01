"""Tests for the hand-written gfx1201 fp8 attention path, through the public `sageattn` API.

fp16, head_dim 128 calls on gfx1201 are served by the prebuilt SK1 kernel by default. These tests
check its accuracy against fp32 SDPA, its edge lengths, its handling of large V values, and that
NHD and HND inputs give the same bits. They are skipped when there is no gfx1201 GPU or no HIP
runtime.
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

# Relative RMS error of the fp8 path against fp32 SDPA on N(0,1) inputs is about 0.03 to 0.055.
MAX_REL_RMS = 0.08

HEAD_DIM = 128


def _sageattn():
    from sageattention import sageattn
    return sageattn


def _rand_qkv(b, h, n, seed=0, layout="HND"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (b, h, n, HEAD_DIM) if layout == "HND" else (b, n, h, HEAD_DIM)
    return [torch.randn(shape, device="cuda", dtype=torch.float16, generator=g) for _ in range(3)]


def _rel_rms(out, ref):
    out = out.float()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _reference(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def test_fp16_hnd_matches_sdpa():
    q, k, v = _rand_qkv(1, 8, 512)
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert out.dtype == torch.float16 and out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, False)) < MAX_REL_RMS


@pytest.mark.parametrize("n", [1, 15, 63, 64, 65, 127, 129, 300, 1000])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_edge_lengths(n, causal, smooth_k):
    q, k, v = _rand_qkv(1, 2, n, seed=n)
    out = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal, smooth_k=smooth_k)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert _rel_rms(out, _reference(q, k, v, causal)) < MAX_REL_RMS


def test_large_v_outlier_stays_finite():
    # All queries are identical and one key is aligned with them, so that key gets almost all of
    # the softmax weight. Its V row has |V| = 20000, far above the e4m3 maximum of 448. A kernel
    # that folds p * amax_v into the fp8 conversion without a clamp overflows here.
    n, outlier = 1024, 300
    q, k, v = _rand_qkv(1, 2, n, seed=1)
    q[:] = q[:, :, :1]
    k[:, :, outlier] = 4 * q[:, :, 0]
    v[:, :, outlier] = 20000.0
    out = _sageattn()(q, k, v, tensor_layout="HND")
    assert (~torch.isfinite(out)).sum().item() == 0


@pytest.mark.parametrize("causal", [False, True])
def test_nhd_equals_hnd_bitwise(causal):
    n = 300
    q, k, v = _rand_qkv(2, 4, n, seed=3)
    out_hnd = _sageattn()(q, k, v, tensor_layout="HND", is_causal=causal)
    qn, kn, vn = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out_nhd = _sageattn()(qn, kn, vn, tensor_layout="NHD", is_causal=causal)
    assert out_nhd.shape == (2, n, 4, HEAD_DIM)
    assert torch.equal(out_nhd.transpose(1, 2), out_hnd)


def test_backend_disabled_by_env_still_works():
    # SAGEATTN_SK1_BACKEND is read at import time, so this has to run in a fresh interpreter.
    code = textwrap.dedent("""
        import torch
        import torch.nn.functional as F
        from sageattention import sageattn
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16, generator=g)
                   for _ in range(3)]
        out = sageattn(q, k, v, tensor_layout="HND")
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        rel = ((out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print("finite=%s rel=%.4f" % (bool(torch.isfinite(out).all()), rel))
    """)
    env = dict(os.environ, SAGEATTN_SK1_BACKEND="0")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    line = [l for l in res.stdout.splitlines() if l.startswith("finite=")][-1]
    assert "finite=True" in line
    assert float(line.split("rel=")[1]) < MAX_REL_RMS
