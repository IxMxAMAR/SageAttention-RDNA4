#!/usr/bin/env python3
"""Rebuild the gfx1201 code objects (`.hsaco`) shipped in `sageattention/sk1_backend/`.

Each object is compiled from one source in `kernels/hip/` with the device-only recipe:

    clang -x hip --offload-arch=gfx1201 --cuda-device-only -O3 -c -o kernel.o kernel.hip
    clang-offload-bundler --type=o --unbundle --targets=hip-amdgcn-amd-amdhsa--gfx1201 \
        --input kernel.o --output kernel.gfx1201.hsaco

`--cuda-device-only` is required: without it the host pass fails with "Can't find declaration for
hipLaunchKernel". `hipcc --genco` is not used because it writes a `clang-offload-bundler` container
(magic `__CLANG_OFFLOAD_BUNDLE__`) rather than a raw ELF, and `hipModuleLoadData` needs the raw ELF.
`-nogpuinc` must not be used either, because `__global__` and `__launch_bounds__` come from the HIP
headers.

No GPU is needed: the build runs with `HIP_VISIBLE_DEVICES=-1` and never initialises the runtime.

The compiler is found, in order, from:

  1. `SK1_CLANG`        path to `clang` (or `clang.exe`) itself
  2. `SK1_ROCM_DEVEL`   root of a ROCm devel tree, i.e. the directory holding `lib/llvm/bin`
  3. the `_rocm_sdk_devel` package of the ROCm Python SDK, in the site-packages of the running
     interpreter and of the user
  4. `ROCM_PATH` / `HIP_PATH`
  5. `clang` on `PATH`

`clang-offload-bundler` and `llvm-objcopy` are taken from the same directory as the compiler.

Usage:

    python tools/build_hsaco.py                  # rebuild all shipped objects in place
    python tools/build_hsaco.py --out-dir build  # write them elsewhere
    python tools/build_hsaco.py --check          # build to a temp dir and compare with the shipped files
    python tools/build_hsaco.py sk1_t4a1s        # only some kernels

The object embeds a symbol `__hip_cuid_<hash>` whose name changes with the source text (even a
comment), so the files are not byte-identical after a comment edit. `--check` therefore compares
the `.text`, `.rodata` and `.note` sections, which hold the machine code, the kernel descriptors and
the resource metadata, and ignores the symbol tables. A different compiler version than the one that
built the shipped objects can also make `--check` report a difference.
"""
import argparse
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(ROOT, "kernels", "hip")
PACKAGE_DIR = os.path.join(ROOT, "sageattention", "sk1_backend")
ARCH = "gfx1201"
SHIPPED = ("sk1_t4a1", "sk1_t4a1s", "sk1_t4a1n", "sk1_t6i", "sk1_t6in", "sk1_t4a1sb", "sk1_t4a1nb",
           "sk1_d64")


def exe(name):
    return name + ".exe" if os.name == "nt" else name


def find_llvm_bin():
    """Directory containing `clang` and `clang-offload-bundler`."""
    clang = os.environ.get("SK1_CLANG")
    if clang:
        if not os.path.isfile(clang):
            raise SystemExit("SK1_CLANG=%s does not exist" % clang)
        return os.path.dirname(os.path.abspath(clang))

    roots = []
    if os.environ.get("SK1_ROCM_DEVEL"):
        roots.append(os.environ["SK1_ROCM_DEVEL"])
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        roots.append(os.path.join(sp, "_rocm_sdk_devel"))
    for var in ("ROCM_PATH", "HIP_PATH"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    for root in roots:
        for sub in (os.path.join("lib", "llvm", "bin"), "bin"):
            d = os.path.join(root, sub)
            if os.path.isfile(os.path.join(d, exe("clang"))):
                return d

    found = shutil.which("clang")
    if found:
        return os.path.dirname(found)
    raise SystemExit("clang not found. Set SK1_CLANG (path to clang) or SK1_ROCM_DEVEL (ROCm devel "
                     "root), or install the ROCm SDK devel package into this Python environment.")


def build(llvm_bin, name, out_dir):
    """Compile kernels/hip/<name>.hip and write <out_dir>/<name>.gfx1201.hsaco."""
    src = os.path.join(KERNEL_DIR, name + ".hip")
    if not os.path.isfile(src):
        raise SystemExit("missing source %s" % src)
    clang = os.path.join(llvm_bin, exe("clang"))
    bundler = os.path.join(llvm_bin, exe("clang-offload-bundler"))
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    os.makedirs(out_dir, exist_ok=True)
    hsaco = os.path.join(out_dir, "%s.%s.hsaco" % (name, ARCH))
    with tempfile.TemporaryDirectory() as tmp:
        obj = os.path.join(tmp, name + ".o")
        steps = [
            [clang, "-x", "hip", "--offload-arch=" + ARCH, "--cuda-device-only", "-O3", "-c",
             "-o", obj, src],
            [bundler, "--type=o", "--unbundle", "--targets=hip-amdgcn-amd-amdhsa--" + ARCH,
             "--input", obj, "--output", hsaco],
        ]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                raise SystemExit("%s failed for %s:\n%s" % (os.path.basename(cmd[0]), name, r.stderr))
    with open(hsaco, "rb") as fh:
        if fh.read(4) != b"\x7fELF":
            raise SystemExit("%s is not a raw ELF code object" % hsaco)
    return hsaco


def code_sections(llvm_bin, path):
    """Bytes of `.text`, `.rodata` (kernel descriptors) and `.note` (metadata) of a code object.

    These carry everything that is executed or read by the runtime. The symbol tables are left out
    because they contain `__hip_cuid_<hash>`, a name that changes with the source text.
    """
    objcopy = os.path.join(llvm_bin, exe("llvm-objcopy"))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sec in (".text", ".rodata", ".note"):
            dump = os.path.join(tmp, "sec.bin")
            r = subprocess.run([objcopy, "--dump-section", sec + "=" + dump, path,
                                os.path.join(tmp, "copy.hsaco")], capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit("llvm-objcopy failed on %s: %s" % (path, r.stderr))
            with open(dump, "rb") as fh:
                out[sec] = fh.read()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kernels", nargs="*", default=list(SHIPPED),
                    help="kernel names under kernels/hip (default: %s)" % " ".join(SHIPPED))
    ap.add_argument("--out-dir", default=PACKAGE_DIR,
                    help="where to write the .hsaco files (default: sageattention/sk1_backend)")
    ap.add_argument("--check", action="store_true",
                    help="build into a temporary directory and compare with the files in "
                         "sageattention/sk1_backend instead of writing")
    args = ap.parse_args()

    llvm_bin = find_llvm_bin()
    print("compiler:", os.path.join(llvm_bin, exe("clang")))
    print(subprocess.run([os.path.join(llvm_bin, exe("clang")), "--version"],
                         capture_output=True, text=True).stdout.splitlines()[0])

    status = 0
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            for name in args.kernels:
                fresh = build(llvm_bin, name, tmp)
                shipped = os.path.join(PACKAGE_DIR, os.path.basename(fresh))
                if not os.path.isfile(shipped):
                    print("%-12s no shipped file to compare" % name)
                    status = 1
                elif code_sections(llvm_bin, fresh) == code_sections(llvm_bin, shipped):
                    print("%-12s .text/.rodata/.note identical to the shipped object" % name)
                else:
                    print("%-12s DIFFERS from the shipped object" % name)
                    status = 1
    else:
        for name in args.kernels:
            out = build(llvm_bin, name, args.out_dir)
            print("%-12s -> %s (%d bytes)" % (name, out, os.path.getsize(out)))
    return status


if __name__ == "__main__":
    sys.exit(main())
