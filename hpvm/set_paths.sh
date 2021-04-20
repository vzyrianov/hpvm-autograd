#!/bin/bash

# These paths can be modified by the HPVM user
CUDA_TOOLKIT_PATH=""  # Set this to the root of your CUDA Installation
if [ -z "$CUDA_TOOLKIT_PATH" ]
then
    echo "ERROR: SET CUDA_TOOLKIT_PATH to the Root of your CUDA Installation"
else
    CUDA_INCLUDE_PATH=$CUDA_TOOLKIT_PATH/include
    CUDA_LIB_PATH=$CUDA_TOOLKIT_PATH/lib64/
    echo "Setting environment paths..."

    # Setting CUDA paths here
    export CUDA_BIN_PATH=$CUDA_TOOLKIT_PATH
    export CUDA_INCLUDE_PATH=$CUDA_INCLUDE_PATH
    export CUDNN_PATH=$CUDA_LIB_PATH
    export LIBRARY_PATH=$CUDA_LIB_PATH:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_LIB_PATH:$LD_LIBRARY_PATH
    echo "Finished setting environment paths!"
fi
