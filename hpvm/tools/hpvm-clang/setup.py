import setuptools

setuptools.setup(
    name="hpvmpy",
    version="1.0",
    author="Yifan Zhao",
    author_email="yifanz16@illinois.edu",
    description="HPVM Python API",
    packages=["hpvmpy"],
    entry_points={
        "console_scripts": [
            "hpvm-clang = hpvmpy:main",
        ],
    },
)
