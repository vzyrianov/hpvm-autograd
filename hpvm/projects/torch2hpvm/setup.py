from setuptools import setup

setup(
    name="torch2hpvm",
    version="1.0",
    description="PyTorch frontend for HPVM",
    author="Yifan Zhao, Yuanjing Shi",
    author_email="yifanz16@illinois.edu, ys26@illinois.edu",
    packages=["torch2hpvm"],
    package_data={"torch2hpvm": ["*.json", "*.cpp.in"]},
    install_requires=[
        "jinja2>=2.11",
        "networkx>=2.5",
        "onnx>=1.8.0",
        # Starting from 1.7.0 PyTorch starts to do some weird optimizations.
        "torch>=1.4,<=1.6",
        "onnx-simplifier>=0.2.27",
    ],
)
