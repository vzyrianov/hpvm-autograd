
from setuptools import setup

setup(
    name='keras_frontend',
    version='0.1',
    description='ApproxHPVM Keras Frontend. Keras -> HPVM Translator',
    author='Hashim',
    author_email='hsharif3@illinois.edu',
    packages=['keras_frontend'],
    install_requires=[
        "tensorflow==1.14",
        "tensorflow-gpu==1.14",
        "keras==2.1.6",
        "scipy==1.1.0",
        "h5py==2.10.0"
    ],
    python_requires="==3.6.*"
)
