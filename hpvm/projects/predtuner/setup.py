import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predtuner",
    version="0.3",
    author="Yifan Zhao",
    author_email="yifanz16@illinois.edu",
    description="A package for predictive and empirical approximation autotuning",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Evan-Zhao/predictive-tuner",
    packages=setuptools.find_packages(),
    package_data={
        "predtuner.approxes": ["default_approx_params.json"]
    },
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.3",
        "networkx>=2.5",
        "torch>=1.5.1",
        "torchvision>=0.6",
        "tqdm>=4.50",
        "pandas>=1.1",
        "jsonpickle>=1.5",
        "argparse>=1.4",
        "wheel",  # Use wheel to build the following:
        "opentuner==0.8.3",  # Must be 0.8.3, they fixed an important bug
        "sqlalchemy==1.3",
    ],
)
