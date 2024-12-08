import os
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension, CppExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name='densematcher',
    version='0.1.2',
    packages=find_packages(include=["densematcher"]),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        'torch',
        'omegaconf',
        'tqdm',
        'scikit-learn',
    ],
    include_package_data=True,
)
