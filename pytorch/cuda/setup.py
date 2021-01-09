from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUDA_HELPER = os.environ.get('CUDA_HELPER', '/usr/local/cuda/samples/common/inc')

setup(
    name='moe_cuda',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda', 
            sources=[
                'moe.cpp',
                'cuda_stream_manager.cpp',
                'moe_cuda_kernel.cu',
                ],
            extra_compile_args={'cxx': ['-I{}'.format(CUDA_HELPER)],
                                'nvcc': ['-I{}'.format(CUDA_HELPER)]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
