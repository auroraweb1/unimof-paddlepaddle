#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import paddle
from paddle.utils.cpp_extension import CUDAExtension, BuildExtension

import os
import subprocess
import sys

from setuptools import find_packages, setup

DISABLE_CUDA_EXTENSION = True
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--enable-cuda-ext":
        DISABLE_CUDA_EXTENSION = False
        continue
    filtered_args.append(arg)
sys.argv = filtered_args


if sys.version_info < (3, 7):
    sys.exit("Sorry, Python >= 3.7 is required for unicore.")


def write_version_py():
    with open(os.path.join("unicore", "version.txt")) as f:
        version = f.read().strip()

    # write version info to unicore/version.py
    with open(os.path.join("unicore", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


if not paddle.is_compiled_with_cuda() and not DISABLE_CUDA_EXTENSION:
    print(
        "\nWarning: Paddle did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, it will cross-compile for Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, _ = get_cuda_bare_metal_version(paddle.sysconfig.get_cuda_home())
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;9.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5"

print("\n\npaddle.version.full_version  = {}\n\n".format(paddle.version.full_version))
PADDLE_MAJOR = int(paddle.version.full_version.split(".")[0])
PADDLE_MINOR = int(paddle.version.full_version.split(".")[1])

if not ((PADDLE_MAJOR >= 2 and PADDLE_MINOR >= 0) or (PADDLE_MAJOR > 2)):
    raise RuntimeError(
        "Requires PaddlePaddle 2.0 or newer.\n"
        + "The latest stable release can be obtained from https://www.paddlepaddle.org.cn/"
    )

cmdclass = {}
ext_modules = []

extras = {}

if not DISABLE_CUDA_EXTENSION:

    def get_cuda_bare_metal_version(cuda_dir):
        raw_output = subprocess.check_output(
            [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]

        return raw_output, bare_metal_major, bare_metal_minor

    def check_cuda_paddle_binary_vs_bare_metal(cuda_dir):
        raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(
            cuda_dir
        )
        paddle_binary_major = paddle.version.cuda().split(".")[0]
        paddle_binary_minor = paddle.version.cuda().split(".")[1]

        print("\nCompiling cuda extensions with")
        print(raw_output + "from " + cuda_dir + "/bin\n")

        if (bare_metal_major != paddle_binary_major) or (
            bare_metal_minor != paddle_binary_minor
        ):
            raise RuntimeError(
                "Cuda extensions are being compiled with a version of Cuda that does "
                + "not match the version used to compile PaddlePaddle binaries.  "
                + "PaddlePaddle binaries were compiled with Cuda {}.\n".format(
                    paddle.version.cuda()
                )
            )

    cmdclass["build_ext"] = BuildExtension

    if paddle.sysconfig.get_cuda_home() is None:
        raise RuntimeError(
            "Nvcc was not found.  Are you sure your environment has nvcc available?"
        )

    # check_cuda_paddle_binary_vs_bare_metal(paddle.sysconfig.get_cuda_home())

    generator_flag = []
    
    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_rounding",
            sources=["csrc/rounding/interface.cpp", "csrc/rounding/fp32_to_bf16.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_multi_tensor",
            sources=[
                "csrc/multi_tensor/interface.cpp",
                "csrc/multi_tensor/multi_tensor_l2norm_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ],
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_adam",
            sources=["csrc/adam/interface.cpp", "csrc/adam/adam_kernel.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                ],
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_softmax_dropout",
            sources=[
                "csrc/softmax_dropout/interface.cpp",
                "csrc/softmax_dropout/softmax_dropout_kernel.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_layernorm",
            sources=["csrc/layernorm/interface.cpp", "csrc/layernorm/layernorm.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_layernorm_backward_gamma_beta",
            sources=[
                "csrc/layernorm/interface_gamma_beta.cpp",
                "csrc/layernorm/layernorm_backward.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-maxrregcount=50",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_rmsnorm",
            sources=["csrc/rmsnorm/interface.cpp", "csrc/rmsnorm/rmsnorm.cu"],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="unicore_fused_rmsnorm_backward_gamma",
            sources=[
                "csrc/rmsnorm/interface_gamma.cpp",
                "csrc/rmsnorm/rmsnorm_backward.cu",
            ],
            include_dirs=[os.path.join(this_dir, "csrc")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ]
                + generator_flag,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-maxrregcount=50",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + generator_flag,
            },
        )
    )

setup(
    name="unicore",
    version=version,
    description="DP Technology's Core AI Framework",
    url="https://github.com/dptech-corp/unicore",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        'numpy; python_version>="3.7"',
        "lmdb",
        "tqdm",
        "paddlepaddle>=2.0.0",
        "ml_collections",
        "scipy",
        "tensorboardX",
        "tokenizers",
        "wandb",
    ],
    packages=find_packages(
        exclude=[
            "build",
            "csrc",
            "examples",
            "examples.*",
            "scripts",
            "scripts.*",
            "tests",
            "tests.*",
        ]
    ),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "unicore-train = unicore_cli.train:cli_main",
        ],
    },
    zip_safe=False,
)