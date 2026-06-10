"""
PyTorch-Wrap - 深度学习快速封装

PyTorch 高级封装框架，提供简洁易用的 API 接口
让深度学习变得更加简单！
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-wrap",
    version="0.1.0",
    author="PyTorch-Wrap Team",
    author_email="pytorch-wrap@example.com",
    description="PyTorch 深度学习封装 - Python 动态神经网络框架 | PyTorch Wrapper for Easy Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/q15004040209-creator/pytorch-wrap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)