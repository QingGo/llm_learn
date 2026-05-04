# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fused_gelu',
    ext_modules=[
        CUDAExtension(
            'fused_gelu',
            ['fused_gelu_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='layernorm_kernel',
    ext_modules=[
        CUDAExtension(
            'layernorm_kernel',
            ['layernorm_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)