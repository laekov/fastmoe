import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cxx_flags = []
ext_libs = []

if os.environ.get('USE_NCCL', '0') == '1':
    cxx_flags.append('-DFMOE_USE_NCCL')
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
        packages=['fmoe', 'fmoe.megatron', 'fmoe.gates'],
        ext_modules=[
            CUDAExtension(
                name='fmoe_cuda', 
                sources=[
                    'cuda/stream_manager.cpp',
                    'cuda/local_exchange.cu',
                    'cuda/balancing.cu',
                    'cuda/global_exchange.cpp',
                    'cuda/parallel_linear.cu',
                    'cuda/fmoe_cuda.cpp',
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
