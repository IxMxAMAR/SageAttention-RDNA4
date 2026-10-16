#!/usr/bin/env python3
"""RX 9060 XT (gfx1200) test probe. Run it with the SAME python.exe that runs ComfyUI:

    <ComfyUI-python.exe> tools/gfx1200_probe.py --out gfx1200_report.json

Use `--out FILE`, not shell redirection. In Windows PowerShell 5.1, `python ... > f.json` merges the
child's stderr into the file and wraps each stderr line in an error record, so one library warning
(Triton prints one on ROCm) makes f.json unreadable. `--out` writes the JSON itself, in UTF-8, and
leaves stdout for the human summary.

It answers, in one process and in this order:

 1. what device and software stack is here;
 2. whether the gfx1200 code objects are in the installed package;
 3. whether the opt-in switch is on, and which object set the package therefore picked;
 4. which code objects actually served the call (the file name of every object it loaded);
 5. whether the output is finite, and how far it is from the package's own fallback path.

It uses random tensors only: no model, no workflow, no image. Paths in the report have your home
directory replaced with `~`, so the file is safe to post.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

#: the loaders' module-level caches, i.e. exactly the objects a call has loaded so far
CACHES = ("_BACKEND", "_BACKEND_N", "_BACKEND_I8", "_BACKEND_I8N", "_BACKEND_B", "_BACKEND_BN")
HOME = os.path.expanduser("~")


def redact(path):
    """Replace the home directory with `~`, so a posted report does not carry a user name."""
    if not path:
        return path
    path = str(path)
    if HOME and HOME != "~" and path.lower().startswith(HOME.lower()):
        return "~" + path[len(HOME):]
    return path


def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def emit(rep, a):
    blob = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(blob + "\n")
        print("wrote the JSON report to %s -- send THAT file back" % a.out)
        print("(stdout carries this human summary; stderr may carry warnings and is not part of it)")
    else:
        print(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print only the JSON report")
    ap.add_argument("--out", metavar="FILE", default=None,
                    help="write the JSON report to FILE (UTF-8). Use this instead of shell "
                         "redirection: PowerShell merges stderr into a redirected file.")
    ap.add_argument("--n", type=int, default=1024, help="sequence length for the served-call check")
    a = ap.parse_args()

    def say(*parts):
        if not a.json:
            print(*parts)

    rep: dict = {"probe": "gfx1200-probe", "python": sys.version.split()[0],
           "platform": platform.platform(), "executable": redact(sys.executable),
           "cwd": redact(os.getcwd())}

    try:
        import torch
        rep["torch"] = torch.__version__
        rep["torch_hip"] = getattr(torch.version, "hip", None)
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        rep["torch_error"] = "%s: %s" % (type(e).__name__, e)

    say("=" * 78)
    say("gfx1200 test probe")
    say("=" * 78)
    say("python      ", rep["python"], "|", rep["executable"])
    say("torch       ", rep.get("torch"), "| HIP", rep.get("torch_hip"))
    say("device      ", rep.get("device_name"), "| cuda_available", rep.get("cuda_available"))

    try:
        import sageattention as SA
        from sageattention import sk1_backend as SB
    except Exception as e:
        rep["sageattention_import_error"] = "%s: %s" % (type(e).__name__, e)
        emit(rep, a)
        return 1

    rep["sageattention_file"] = redact(os.path.abspath(SA.__file__))
    rep["sageattention_version"] = getattr(SA, "__version__", None)
    try:
        from sageattention import _version as V
        rep["build_id"] = getattr(V, "BUILD_ID", None)
    except Exception:
        pass

    # A tester may have an older package installed; say so in words rather than with a traceback.
    rep["env_SAGEATTN_SK1_GFX1200"] = os.environ.get("SAGEATTN_SK1_GFX1200")
    rep["SK1_GFX1200"] = bool(getattr(SB, "SK1_GFX1200", False))
    rep["has_gfx1200_routing"] = all(hasattr(SB, n) for n in
                                     ("selected_arch", "arch_supported", "gfx1200_status",
                                      "SUPPORTED_ARCHS"))
    rep["supported_archs"] = list(getattr(SB, "SUPPORTED_ARCHS", ()))
    rep["device_arch"] = SB.device_arch()
    rep["selected_arch"] = SB.selected_arch() if rep["has_gfx1200_routing"] else None
    rep["arch_supported"] = bool(SB.arch_supported()) if rep["has_gfx1200_routing"] else None
    if rep["has_gfx1200_routing"]:
        st = SB.gfx1200_status()
        rep["gfx1200_status"] = dict(switch=st["switch"], default_on=st["default_on"],
                                     base_arch=st.get("base_arch"),
                                     selected_now=st.get("selected_now"),
                                     objects={k: {"file": v["file"], "present": v["present"],
                                                  "size": v["size"], "sha256": v["sha256"]}
                                              for k, v in st["objects"].items()})
    else:
        rep["gfx1200_status"] = None
    rep["resolved_object_names"] = {
        n: getattr(SB, n, None) for n in
        ("SK1_HSACO_NAME", "SK1_HSACO_NAME_S", "SK1_HSACO_NAME_N", "SK1_HSACO_NAME_I8",
         "SK1_HSACO_NAME_I8N", "SK1_HSACO_NAME_B", "SK1_HSACO_NAME_BN")}

    say("-" * 78)
    say("device arch      ", rep["device_arch"])
    say("package          ", rep["sageattention_file"], "| version", rep["sageattention_version"])
    if not rep["has_gfx1200_routing"]:
        say("gfx1200 routing  ", "ABSENT -- the installed package predates the gfx1200 opt-in;")
        say("                 ", "it does not carry the gfx1200 objects")
    say("selected arch    ", rep["selected_arch"], "(None = the package will refuse)")
    say("switch           ", "ON" if rep["SK1_GFX1200"] else "off",
        "| SAGEATTN_SK1_GFX1200 =", rep["env_SAGEATTN_SK1_GFX1200"])
    if rep["gfx1200_status"]:
        miss = [k for k, v in rep["gfx1200_status"]["objects"].items() if not v["present"]]
        say("gfx1200 objects  ", "ALL PRESENT" if not miss else "MISSING: %s" % ", ".join(miss))
    say("would load       ", rep["resolved_object_names"]["SK1_HSACO_NAME_S"])

    if not rep.get("cuda_available"):
        rep["served"] = None
        rep["note"] = "no GPU visible to torch; nothing was called"
    else:
        import torch
        torch.manual_seed(1234)
        B, H, N, D = 1, 8, a.n, 128
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5 for _ in range(3))
        out = SA.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()
        off = SA.sageattn(q, k, v, tensor_layout="HND", is_causal=False, sk1_backend=False)
        torch.cuda.synchronize()
        served = {}
        for name in CACHES:
            mod = getattr(SB, name, None)
            if mod is not None:
                p = getattr(mod, "hsaco_path", None)
                served[name] = os.path.basename(p) if p else str(mod)[:80]
        rep["served"] = dict(objects=served, arch_loaded=getattr(SB, "_ARCH_LOADED", None),
                             out_shape=list(out.shape), out_dtype=str(out.dtype),
                             out_finite=bool(torch.isfinite(out).all().item()),
                             out_absmax=float(out.abs().max().item()),
                             n_nonfinite=int((~torch.isfinite(out)).sum().item()),
                             max_abs_diff_vs_fallback=float((out.float() - off.float()).abs().max().item()),
                             fallback_finite=bool(torch.isfinite(off).all().item()))
        say("-" * 78)
        say("served by        ", served if served else "NOTHING -- the package fell back")
        say("output           ", "finite" if rep["served"]["out_finite"] else "HAS NON-FINITE VALUES",
            "| absmax", rep["served"]["out_absmax"])
        say("vs fallback      ", "max |diff| = %.6g" % rep["served"]["max_abs_diff_vs_fallback"])
        if not served:
            say("  !! no code object was loaded: the call did NOT use the prebuilt kernels")
        elif all(n.endswith("_1200.hsaco") for n in served.values()) and rep["device_arch"] == "gfx1200":
            say("  OK: the gfx1200 objects served this call")
        elif all(n.endswith(".gfx1201.hsaco") for n in served.values()):
            say("  OK: the gfx1201 objects served this call")

    say("-" * 78)
    emit(rep, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
