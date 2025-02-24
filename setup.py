from setuptools import setup, find_packages

setup(
    name="firconv",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='Frog',  # 作者信息
    description='A useful module for building a FIR (Finite Impulse Response) filter in Deep Learning Neural Networks, implemented using PyTorch. This package provides an easy-to-use interface to integrate pre-defined filters into your neural network or train the filter coefficients as learnable parameters. It allows you to observe how the neural network captures and adapts to specific frequency bands during training, making it a powerful tool for signal processing tasks in deep learning applications.'
)
