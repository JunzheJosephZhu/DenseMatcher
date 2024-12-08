from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension, CppExtension

setup(
    name='featup',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torchmetrics',
    ],
    ext_modules=[
        CUDAExtension(
            'adaptive_conv_cuda_impl',
            [
                'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
                'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
            ]),
        CppExtension(
            'adaptive_conv_cpp_impl',
            ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
            undef_macros=["NDEBUG"]),

    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)