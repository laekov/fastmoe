import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

cxx_flags = []
ext_libs = []

authors = [
        'Jiaao He', 
        'Jiezhong Qiu', 
        'Aohan Zeng', 
        'Tiago Antunes', 
        'Jinjun Peng', 
        'Qin Li',
]

is_rocm_pytorch = False
if torch.__version__ >= '1.5':
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

if os.environ.get('USE_NCCL', '1') == '1':
    cxx_flags.append('-DFMOE_USE_NCCL')
    cxx_flags.append('-DUSE_C10D_NCCL')
    if is_rocm_pytorch:
        ext_libs.append('rccl')
    else:
        ext_libs.append('nccl')

if is_rocm_pytorch:
    define_macros=[('FMOE_USE_HIP', None)]
else:
    define_macros=[]


if __name__ == '__main__':
    setuptools.setup(
        name='fastmoe',
        version='0.3.0',
        description='An efficient Mixture-of-Experts system for PyTorch',
        author=', '.join(authors),
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
                define_macros=define_macros,
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
