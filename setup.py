"""
Build script for the AMD gfx12 SageAttention wheel.

This is SageAttention PR #368's ROCm build, reduced to the files this port actually
ships, with four packaging changes:

  1. No hardcoded C++ standard. torch picks `-std=c++20`, which is what the gfx1201
     build uses. SAGEATTN_CXX_STD still forces one.
  2. `_GLIBCXX_USE_CXX11_ABI` is propagated from torch to both cxx and hipcc.
     Removing it is an ABI break.
  3. A missing gfx12 target is a hard error, not a warning. With PYTORCH_ROCM_ARCH
     unset, a warning would produce a wheel with no gfx12 extension, which silently
     falls back to PyTorch attention.
  4. The distribution version carries a local identifier so the wheel is
     distinguishable from the upstream/PyPI artifact of the same name.

Two options exist for packaging already-built extensions:

  * `SAGEATTN_SKIP_CUDA_BUILD=1` skips extension configuration entirely, so
    `bdist_wheel` can package `.pyd` files already present in `sageattention/`
    without recompiling.
  * `--no-build-ext` additionally clears `ext_modules`, so `bdist_wheel` does not
    try to invoke `build_ext` at all.

Build the wheel from source (needs a ROCm torch and the ROCm SDK in the active
environment; setup.py imports torch, so build isolation must be off):

    set PYTORCH_ROCM_ARCH=gfx1201
    pip wheel . --no-build-isolation --no-deps
    (or: python setup.py bdist_wheel)

Package already-built extensions without compiling:

    python setup.py --no-build-ext bdist_wheel

The compile takes tens of minutes. The `.hsaco` code objects under
`sageattention/sk1_backend/` are not built here; see `tools/build_hsaco.py`.
"""

import os
import sys
import threading
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages

VERSION = "2.2.0+amd.gfx12.2"
DIST_NAME = "sageattention"
TARGET_TORCH = "2.13.0+rocm10.0.0"
TARGET_ROCM = "7.15.26333"
TARGET_ARCH = "gfx1201"

NO_BUILD_EXT = "--no-build-ext" in sys.argv
if NO_BUILD_EXT:
    sys.argv.remove("--no-build-ext")

# Skip extension configuration when packaging prebuilt binaries, in CI, or when
# explicitly requested.
SKIP_CUDA_BUILD = (
    NO_BUILD_EXT
    or os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}


def rocm_sdk_path(which):
    try:
        import subprocess

        return subprocess.check_output(["rocm-sdk", "path", f"--{which}"], text=True).strip()
    except Exception:
        return None


def rocm_arches(torch):
    arch_env = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if arch_env:
        return [a.strip().split(":", 1)[0]
                for a in arch_env.replace(";", ",").split(",") if a.strip()]

    archs = []
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_idx)
            arch = getattr(props, "gcnArchName", "")
            if arch:
                archs.append(arch.split(":", 1)[0])
    return archs


def unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def configure_rocm(default_rocm_home):
    sdk_root = rocm_sdk_path("root")
    sdk_bin = rocm_sdk_path("bin")
    rocm_home = sdk_root or default_rocm_home or os.getenv("ROCM_HOME")
    if not rocm_home:
        raise RuntimeError("Cannot find ROCm. Activate the ROCm Python environment.")

    os.environ["ROCM_HOME"] = rocm_home
    if os.name == "nt":
        os.environ.setdefault("CC", "clang-cl")
        os.environ.setdefault("CXX", "clang-cl")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    path_parts = [
        os.path.join(rocm_home, "lib", "llvm", "bin"),
        os.path.join(rocm_home, "bin"),
        sdk_bin,
    ]
    os.environ["PATH"] = os.pathsep.join(unique_paths(path_parts) + [os.environ.get("PATH", "")])
    return rocm_home


if not SKIP_CUDA_BUILD:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "setup.py needs the ROCm build of torch importable to build the extensions. "
            "Install it first and build with `pip wheel . --no-build-isolation`."
        ) from exc

    if torch.version.hip is not None:
        import torch.utils.cpp_extension as cpp_extension
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

        ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
        rocm_home = configure_rocm(ROCM_HOME)
        cpp_extension.ROCM_HOME = rocm_home
        amd_arches = rocm_arches(torch) or [TARGET_ARCH]
        os.environ.setdefault("PYTORCH_ROCM_ARCH", ";".join(amd_arches))
        print(f"Target AMD GPU architectures: {amd_arches}")

        # torch appends the standard its own headers need, but only when the
        # extension sets none, so hardcoding one pins it. SAGEATTN_CXX_STD forces
        # a value; leaving it unset lets torch supply -std=c++20.
        cxx_std = os.getenv("SAGEATTN_CXX_STD", "").strip()
        msvc_std = [f"/std:{cxx_std}"] if cxx_std else []
        clang_std = [f"-std={cxx_std}"] if cxx_std else []

        if os.name == "nt":
            CXX_FLAGS = ["/O2", *msvc_std, f"/D_GLIBCXX_USE_CXX11_ABI={ABI}", "/DENABLE_BF16"]
        else:
            CXX_FLAGS = ["-O3", *clang_std, f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-DENABLE_BF16"]

        HIP_FLAGS = [
            "-O3",
            *clang_std,
            "-ffast-math",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
            "-mllvm",
            "--lsr-drop-solution=1",
            "-mllvm",
            "-enable-post-misched=1",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
        ]
        for arch in amd_arches:
            HIP_FLAGS.append(f"--offload-arch={arch}")
        HIP_FLAGS.append(f"--rocm-path={rocm_home}")
        rocm_device_lib_path = os.path.join(rocm_home, "lib", "llvm", "amdgcn", "bitcode")
        if os.path.isdir(rocm_device_lib_path):
            HIP_FLAGS.append(f"--rocm-device-lib-path={rocm_device_lib_path}")

        cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
        if cxx_append:
            CXX_FLAGS += cxx_append.split()
        nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
        if nvcc_append:
            HIP_FLAGS += nvcc_append.split()
        hipcc_append = os.getenv("HIPCC_APPEND_FLAGS", "").strip()
        if hipcc_append:
            HIP_FLAGS += hipcc_append.split()

        include_dirs = unique_paths([os.path.join(rocm_home, "include")])

        # A missing gfx12 target is an error, not a warning.
        if not any(arch.startswith("gfx12") for arch in amd_arches):
            raise RuntimeError(
                f"ROCm build detected but no gfx12 architecture was selected "
                f"(got {amd_arches}). The gfx12 native attention extension would "
                f"be omitted and the wheel would silently degrade to PyTorch "
                f"attention. Set PYTORCH_ROCM_ARCH={TARGET_ARCH}."
            )

        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_gfx12_native",
                sources=[
                    "csrc/qattn/pybind_gfx12_native.cpp",
                    "csrc/qattn/qk_int_sv_gfx12_native_aux.cu",
                    "csrc/qattn/qk_int_sv_gfx12_native_prepare.cu",
                    "csrc/qattn/qk_int_sv_gfx12_native_attn_f16.cu",
                    "csrc/qattn/qk_int_sv_gfx12_native_attn_fp8.cu",
                    "csrc/qattn/qk_int_sv_gfx12_native_rawq_fp8.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": HIP_FLAGS},
            )
        )

        ext_modules.append(
            CUDAExtension(
                name="sageattention._fused",
                sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
                include_dirs=include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": HIP_FLAGS},
            )
        )

        parallel = None
        if "EXT_PARALLEL" in os.environ:
            parallel = int(os.getenv("EXT_PARALLEL"))
        if parallel is None and "MAX_JOBS" in os.environ:
            parallel = int(os.getenv("MAX_JOBS"))
        if parallel is None:
            parallel = 4
        os.environ.setdefault("MAX_JOBS", "32")

        class BuildExtensionSeparateDir(BuildExtension):
            build_extension_patch_lock = threading.Lock()
            thread_ext_name_map = {}

            def finalize_options(self):
                if parallel is not None:
                    self.parallel = parallel
                super().finalize_options()

            def build_extension(self, ext):
                with self.build_extension_patch_lock:
                    if not getattr(self.compiler, "_compile_separate_output_dir", False):
                        compile_orig = self.compiler.compile

                        def compile_new(*args, **kwargs):
                            return compile_orig(*args, **{
                                **kwargs,
                                "output_dir": os.path.join(
                                    kwargs["output_dir"],
                                    self.thread_ext_name_map[threading.current_thread().ident]),
                            })
                        self.compiler.compile = compile_new
                        self.compiler._compile_separate_output_dir = True
                self.thread_ext_name_map[threading.current_thread().ident] = ext.name
                return super().build_extension(ext)

        cmdclass = {"build_ext": BuildExtensionSeparateDir} if ext_modules else {}
    elif not NO_BUILD_EXT and os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() not in {"1", "TRUE", "YES"}:
        raise RuntimeError(
            "This setup.py builds the AMD/gfx12 wheel only; the installed torch "
            "is not a ROCm build (torch.version.hip is None). Build upstream "
            "SageAttention for CUDA instead."
        )

if NO_BUILD_EXT:
    # `root_is_pure` is derived from `distribution.has_ext_modules()`, which is
    # just `bool(ext_modules)`. With ext_modules emptied, setuptools would tag the
    # wheel py3-none-any and drop the payload under `.data/purelib/`. A dummy
    # Extension keeps the distribution marked as platform-specific so the wheel is
    # tagged cp312-cp312-win_amd64 and the .pyd files sit at the wheel root.
    # `build_ext` is replaced with a no-op so the placeholder is never compiled;
    # the prebuilt `.pyd` files ship via package_data.
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext as _build_ext

    ext_modules = [Extension("sageattention._prebuilt_placeholder", sources=[])]

    class NoOpBuildExt(_build_ext):
        def run(self):
            pass

        def build_extensions(self):
            pass

    cmdclass = {"build_ext": NoOpBuildExt}

# The wheel contains compiled .pyd files, so it must be tagged platform-specific
# (cp312-cp312-win_amd64) rather than py3-none-any.
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

    class BdistWheelPlatform(_bdist_wheel):
        def get_tag(self):
            python, abi, plat = super().get_tag()
            if python == "py3":
                python, abi = "cp312", "cp312"
            return python, abi, plat

    cmdclass = {**cmdclass, "bdist_wheel": BdistWheelPlatform}
except ImportError:
    warnings.warn("wheel support missing; bdist_wheel may produce a pure-python tag")

setup(
    name=DIST_NAME,
    version=VERSION,
    author="SageAttention team; AMD gfx12 port",
    license="Apache 2.0 License",
    description="Accurate and efficient plug-and-play low-bit attention (AMD gfx12/RDNA4 native port).",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/SageAttention",
    packages=find_packages(include=["sageattention", "sageattention.*"]),
    package_data={
        "sageattention": ["*.pyd"],
        # The prebuilt gfx1201 code objects ship as package data, with the ctypes loader and the
        # fused quantiser they need. No compiler or ROCm toolchain is required to install them;
        # the HIP runtime DLL is resolved at runtime (sageattention/sk1_backend/sk1_loader.py).
        "sageattention.sk1_backend": ["*.hsaco", "*.py"],
    },
    include_package_data=True,
    python_requires=">=3.12,<3.13",
    install_requires=[f"torch=={TARGET_TORCH}"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
