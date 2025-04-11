import os
import torch
from torch.utils.cpp_extension import load, CUDA_HOME
import subprocess
from scripts.utils import get_nvidia_cc


def get_cuda_bare_metal_version(cuda_dir):
    if cuda_dir == None or torch.version.cuda == None:
        print("CUDA is not found, cpu version is installed")
        return None, -1, 0
    else:
        raw_output = subprocess.check_output(
            [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]

        return raw_output, bare_metal_major, bare_metal_minor


# Version dependent macros
version_dependent_macros = [
    "-DVERSION_GE_1_1",
    "-DVERSION_GE_1_3",
    "-DVERSION_GE_1_5",
]

# CUDA flags
extra_cuda_flags = [
    "-std=c++17",
    "-maxrregcount=50",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

# Compute capabilities setup
compute_capabilities = set(
    [
        (5, 2),  # Titan X
        (6, 1),  # GeForce 1000-series
    ]
)

compute_capabilities.add((7, 0))
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    compute_capabilities.add((8, 0))

# Get specific GPU compute capability if available
compute_capability, _ = get_nvidia_cc()
if compute_capability is not None:
    compute_capabilities = set([compute_capability])

# Add compute capability flags
cc_flag = []
for major, minor in list(compute_capabilities):
    cc_flag.extend(
        [
            "-gencode",
            f"arch=compute_{major}{minor},code=sm_{major}{minor}",
        ]
    )

extra_cuda_flags += cc_flag

# Source files and include directories
sources = [
    "openfold/utils/kernel/csrc/softmax_cuda.cpp",
]

include_dirs = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "openfold/utils/kernel/csrc/",
    )
]

if bare_metal_major != -1:  # If CUDA is available
    sources.append("openfold/utils/kernel/csrc/softmax_cuda_kernel.cu")
    extra_compile_args = {
        "cxx": ["-O3"] + version_dependent_macros,
        "nvcc": (
            ["-O3", "--use_fast_math"] + version_dependent_macros + extra_cuda_flags
        ),
    }
else:  # CPU only
    sources.append("openfold/utils/kernel/csrc/softmax_cuda_stub.cpp")
    extra_compile_args = {"cxx": ["-O3"]}

# Load and compile the extension
attn_core_inplace = load(
    name="attn_core_inplace_cuda",
    sources=sources,
    extra_cflags=(
        ["-O3"] + version_dependent_macros if bare_metal_major != -1 else ["-O3"]
    ),
    extra_cuda_cflags=(
        (["-O3", "--use_fast_math"] + version_dependent_macros + extra_cuda_flags)
        if bare_metal_major != -1
        else None
    ),
    extra_include_paths=[
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "openfold/utils/kernel/csrc/",
        )
    ],
    verbose=True,
    with_cuda=bare_metal_major != -1,
)

print("Extension successfully built!")
