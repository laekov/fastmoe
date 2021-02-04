import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUDA_HELPER = os.environ.get('CUDA_HELPER', '/usr/local/cuda/samples/common/inc')
cxx_flags = [
        '-I{}'.format(CUDA_HELPER)
        ]
if os.environ.get('USE_NCCL', '0') == '1':
    cxx_flags.append('-DMOE_USE_NCCL')

if __name__ == '__main__':
    setuptools.setup(
        name='fmoe',
        packages=['fmoe'],
        ext_modules=[
            CUDAExtension(
                name='fmoe_cuda', 
                sources=[
                    'cuda/moe.cpp',
                    'cuda/cuda_stream_manager.cpp',
                    'cuda/moe_compute_kernel.cu',
                    'cuda/moe_comm_kernel.cu',
                    'cuda/moe_fused_kernel.cu',
                    ],
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': cxx_flags
                    }
                )
            ],
        version='0.0.2',
        cmdclass={
            'build_ext': BuildExtension
        })
