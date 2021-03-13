import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cxx_flags = []
ext_libs = []

if os.environ.get('USE_NCCL', '0') == '1':
    cxx_flags.append('-DMOE_USE_NCCL')
    ext_libs.append('nccl')


if __name__ == '__main__':
    setuptools.setup(
        name='fastmoe',
        version='0.1.2',
        description='An efficient Mixture-of-Experts system for PyTorch',
        author='Jiaao He, Jiezhong Qiu and Aohan Zeng',
        author_email='hja20@mails.tsinghua.edu.cn',
        license='Apache-2',
        url='https://github.com/laekov/fastmoe',
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
                    },
                libraries=ext_libs
                )
            ],
        cmdclass={
            'build_ext': BuildExtension
        })
