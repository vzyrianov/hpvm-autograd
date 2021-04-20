ARG IMAGE_NAME=nvidia/cuda
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install dependencies: python 3.6, curl, git, libboost
RUN apt-get update && apt-get install -y --no-install-recommends python3 curl git libboost-dev

# Install cmake
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh -o cmake.sh && \
    bash ./cmake.sh --skip-license --prefix=/usr && rm cmake.sh

# Install conda
RUN curl https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -o anaconda.sh && \
    bash anaconda.sh -b && rm anaconda.sh

# Set PATH to include conda
ENV PATH="/root/anaconda3/bin:${PATH}"

# Send conda env spec into container
COPY . /root/hpvm/

# Create conda env named hpvm based on spec
RUN conda env create -n hpvm -f /root/hpvm/hpvm/env.yaml
