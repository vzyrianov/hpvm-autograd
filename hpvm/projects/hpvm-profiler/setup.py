import setuptools

setuptools.setup(
    name="hpvm_profiler",
    version="0.1",
    author="Akash Kothari, Yifan Zhao",
    author_email="akashk4@illinois.edu, yifanz16@illinois.edu",
    description="A package for profiling of HPVM approximation configurations",
    packages=["hpvm_profiler"],
    install_requires=["numpy>=1.19", "matplotlib>=3"],
)
