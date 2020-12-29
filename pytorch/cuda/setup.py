from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='moe_cuda',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda', 
            sources=[
                'moe.cpp',
                'moe_cuda_kernel.cu',
                ],
            extra_compile_args={'cxx': ['-I/usr/local/cuda/samples/common/inc'],
                                'nvcc': ['-I/usr/local/cuda/samples/common/inc']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })