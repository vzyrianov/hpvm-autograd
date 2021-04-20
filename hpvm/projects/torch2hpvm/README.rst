PyTorch Frontend for HPVM
=========================

``torch2hpvm`` is a PyTorch frontend for HPVM. It provides a set of API that

* Generates a PyTorch ``module`` into HPVM-C code;
* Exports a PyTorch dataset to ApproxHPVM dataset format;
* Compiles the generated code into binary by invoking HPVM automatically.

Installation
------------

``pip3`` is the recommended package manager (also available within ``conda``).
Using ``pip3``:

.. code-block:: bash

   pip3 install -e ./

Getting Started
---------------

Let's look at an example that uses DNNs and weights pre-shipped with HPVM.
This is found at ``hpvm/test/dnn_benchmarks/pytorch/test_frontend.py``.
**Note** that below we'll be working under directory ``hpvm/test/dnn_benchmarks/pytorch``.

We'll be generating ResNet-18 into an HPVM-compiled binary.
First, prepare 2 datasets for autotuning and testing.

.. code-block:: python

   from torch2hpvm import BinDataset
   from pathlib import Path

   data_dir = Path(__file__).parent / "../model_params/resnet18_cifar10"
   dataset_shape = 5000, 3, 32, 32
   tuneset = BinDataset(data_dir / "tune_input.bin", data_dir / "tune_labels.bin", dataset_shape)
   testset = BinDataset(data_dir / "test_input.bin", data_dir / "test_labels.bin", dataset_shape)

``BinDataset`` is a dataset created over files of ApproxHPVM dataset format.
Any instance ``torch.utils.data.Dataset`` can be used here.

**Note** that each ``module`` is bound to 2 datasets: a "tune" and a "test" set.
The generated binary accepts an argument to be either the string "tune" or "test",
and performs inference over a dataset accordingly.
This is because the dataset can contain arbitrary Python code which cannot yet be exported into HPVM-C;
instead the frontend has to export some predefined datasets for the model to use.

Create a DNN ``module`` and load the checkpoint:

.. code-block:: python

   import torch
   from torch.nn import Module
   import dnn  # Defined at `hpvm/test/dnn_benchmarks/pytorch`

   model: Module = dnn.ResNet18()
   checkpoint = Path(__file__).parent / "../model_params/resnet18_cifar10.pth.tar"
   model.load_state_dict(torch.load(checkpoint))

Any ``torch.nn.Module`` can be similarly used,
as long as they only contain the tensor operators supported in HPVM
(see "Supported Operators" and TODOs (2)).

Now we are ready to export the model. The main functioning class of ``torch2hpvm`` is ``ModelExporter``:

.. code-block:: python

   from torch2hpvm import ModelExporter

   output_dir = Path("./resnet18_hpvm")
   build_dir = output_dir / "build"
   target_binary = build_dir / "resnet18"
   batch_size = 500
   conf_file = "" # Change this to point to your configuration file.
   exporter = ModelExporter(model, tuneset, testset, output_dir, config_file=conf_file)
   exporter.generate(batch_size=batch_size).compile(target_binary, build_dir)

``output_dir``, ``build_dir``, and ``target_binary`` define the folder for code generation, compilation,
and path to the compiled binary respectively.
``batch_size`` is the batch size the binary uses during inference.

**Note** that ``conf_file`` is the path to an HPVM approximation configuration file.
This file decides what approximation the binary will use during inference.
This path is hardcoded into the binary and is only read when the binary starts,
so it's fine to have ``conf_file`` point to a non-existing path.
An example can be found at ``test/dnn_benchmarks/hpvm-c/benchmarks/resnet18_cifar10/data/tuner_confs.txt``.

Supported Operators
-------------------

Any builtin and custom PyTorch ``Module`` are supported
*as long as* the generated ONNX model consists of only the following operators
when the Module is exported into ONNX:

.. list-table::
   :header-rows: 1

   * - Convolution
     - Linear
     - Pooling
     - Pointwise
     - Other
   * - Conv
     - MatMul
     - GlobalAveragePool
     - BatchNormalization
     - Flatten
   * - 
     - Gemm
     - AveragePool
     - Relu
     - Softmax
   * - 
     - 
     - MaxPool
     - Tanh
     - Identity
   * - 
     - 
     - 
     - 
     - Pad
   * - 
     - 
     - 
     - 
     - Add


This choice of operators is largely constrained by backend (tensor_runtime) supports.

TODOs
-----

#. Optionally insert a Python-C interface in the generated binary to
   call back into a Dataset class and read the data.

   * Needs pybind11, hardcoding of Python environment, and some fiddling with import mechanism.

#. Expand the list of operators supported in the frontend.

   * Most ideally, create a high-level description of operators that can tie
     HPVM-C intrinsics and the frontend list of operators together.
