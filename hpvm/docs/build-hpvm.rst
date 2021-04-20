Building HPVM
===============

Dependencies
------------

* The following components are mandatory for building HPVM:

   * GCC (>=5.1)

   * CMake (>=3.17)

   * GNU Make (>=3.79) or Ninja (>=1.10)

   * Python (==3.6) with pip (>=20)

      * Python must be strictly 3.6 (any subversion from 3.6.0 to 3.6.13).
        This is needed by some Python packages in HPVM.
      
      * If you choose to not install these packages, then any Python >= 3.6 will work.
        See :ref:`how to skip installing Python packages in the installer <skip-pypkg>`.

* OpenCL (>=1.0.0) is required for compiling HPVM-C code on GPU; otherwise, only CPU is available.

* The following components are required to build support for Tensor-domain applications
  introduced in `ApproxHPVM <https://dl.acm.org/doi/10.1145/3360612>`_:

   * CUDA (>=9.0, <=10.2) with CUDNN 7

      * CUDNN 7 is unsupported beyond CUDA 10.2 (starting from CUDA 11)
   
   * OpenMP (>= 4.0)

      * GCC comes with OpenMP support; OpenMP-4.0 is supported by GCC-4.9 onward.
        see `here <https://gcc.gnu.org/wiki/openmp>`__ for the OpenMP version supported by each GCC version.
 
   * In addition, each version of CUDA-nvcc requires GCC to be not newer than a certain version.
     See `here <https://gist.github.com/ax3l/9489132>`__ for the support matrix.


Python Environment
^^^^^^^^^^^^^^^^^^

It is strongly recommended to use some Python virtual environment,
as HPVM will install a few Python packages during this installation process.

* Some HPVM Python packages contains executables. If you don't use a virtual environment,
  these executables are installed to your local ``bin`` directory, usually ``$HOME/.local/bin``.
  Please ensure this directory is in your `$PATH` variable.
  Below it is assumed that these executables are visible through `$PATH`.

If you use Anaconda for package management,
we provide a conda environment file that covers all Python and package requirements
(``hpvm/env.yaml`` can be found in the repository):

.. code-block:: bash

   conda env create -n hpvm -f hpvm/env.yaml

This creates the conda environment ``hpvm``.
If you use this method, remember to activate the environment each time you enter a bash shell:

.. code-block:: bash

   conda activate hpvm

Supported Architectures
-----------------------

Supported/tested CPU architectures:

* Intel Xeon E5-2640
* Intel Xeon W-2135
* ARM Cortex A-57

Supported/tested GPU architectures for OpenCL backend:

* Nvidia Quadro P1000
* Nvidia GeForce GTX 1080

Supported/tested GPU architectures for Tensor Backend:

* Nvidia Jetson TX2
* Nvidia GeForce GTX 1080

HPVM has not been tested on other architectures,
but it is expected to work on CPUs supported by the LLVM Backend
and GPUs supported by OpenCL (Intel, AMD, etc.).

**NOTE**: Approximations are tuned for Jetson TX2 and same speedups may not exist for other architectures.


Installing from Source
----------------------

Checkout HPVM and go to directory ``./hpvm`` under project root:

.. code-block:: shell

   git clone --recursive -b main https://gitlab.engr.illinois.edu/llvm/hpvm-release.git
   cd hpvm/

If you have already cloned the repository without using ``--recursive``,
the directory ``hpvm/projects/predtuner`` should be empty,
which can be fixed with ``git submodule update --recursive --init``.

HPVM needs to be able to find CUDA.
If CUDA is installed in your system's ``$PATH`` (e.g. if it was installed at the default location),
HPVM can find CUDA automatically.

Use HPVM installer script to download extra components, configure and build HPVM:

.. code-block:: shell

   ./install.sh

* Without arguments, this script will interactively prompt you for some parameters.
  Alternatively, use ``./install.sh -h`` for a list of available arguments
  and pass arguments as required.

* ``./install.sh`` supports `Ninja <https://ninja-build.org/>`_,
  a substitute of Make that is considered to build faster on many IO-bottlenecked devices.
  Passing ``--ninja`` to the installer tells it to use Ninja instead of Make.

* ``./install.sh`` can relay additional arguments to CMake, but the dash must be dropped
  regardless of using prompt or CLI arguments.
  For example, 

  .. code-block:: shell

   ./install.sh -j32 DCMAKE_BUILD_TYPE=Release

  will compile HPVM with 32 threads in Release mode; similarly, inputting
  ``DCMAKE_BUILD_TYPE=Release`` to the prompt will also send ``-DCMAKE_BUILD_TYPE=Release``
  to CMake which gives a build in Release mode.

After configuring HPVM,
the installer will also compile HPVM by default, which you can opt out of.
(You can see this option in both the prompt and the ``-h`` help menu.)
If you do so, follow the next section "Manually Build HPVM" to manually compile HPVM,
and "Benchmarks and Tests" to manually run test cases if you wish so.
Otherwise, you can skip the next 2 sections.

How Does the Installer Work
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HPVM installer performs the following tasks:

* downloads and decompresses LLVM and Clang into `hpvm/llvm`,

* links HPVM source code into ``hpvm/llvm/tools/hpvm``,

* downloads DNN model parameters to ``test/dnn_benchmarks/model_params`` (this step is optional -- you can opt out of it),

* installs a few Python packages: the PyTorch frontend, the Keras frontend, the predictive tuner,
  and the HPVM profiler, (this step is optional),

* builds the entire HPVM which provides `hpvm-clang`, HPVM's main compilation interface,

  * The build system builds HPVM, creates a Python package `hpvmpy` (which provides the binary `hpvm-clang`)
    *on the fly*, and installs it to your current Python environment.

* and finally, builds and runs some tests if you explicitly require so.

  * While running tests is recommended, it is not turned on by default as it is very time-consuming.

.. _skip-pypkg:

Skipping Python Package installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are installing HPVM on a "target" device which is just used for
:ref:`profiling <target-profiling>`,
you may not need to install the frontend and the tuner packages.
These packages also have Python version requirement and package dependencies
that may be hard to meet on some devices, especially edge computing devices with ARM CPUs.

You can instead skip the installation by either passing ``--no-pypkg`` flag to
the installer, or answering yes ("y") when it prompt the following:

.. code-block:: text

   Install HPVM Python Packages (recommended)? [y/n]

In this case, any Python >= 3.6 will work.

TroubleShooting
^^^^^^^^^^^^^^^

If CMake did not find your CUDA, some environment variables will help it:

* ``CUDA_TOOLKIT_PATH`` --- Path to the CUDA toolkit
* ``CUDA_INCLUDE_PATH`` --- Path to the CUDA headers
* ``CUDA_LIB_PATH`` --- Path to CUDA libraries

You can use ``set_paths.sh`` for this purpose: modify the values of these variables
in ``set_paths.sh`` according to your system, and source the script:

.. code-block:: shell

   source set_paths.sh

Manually Build HPVM
-------------------

Alternatively, you can manually build HPVM with CMake.
Please note that in this case,
the installer script still *must* be executed to obtain some required components,
but without the build step.
In current directory (``hpvm/``), do

.. code-block:: shell

   mkdir build
   cd build
   cmake ../llvm [options]

Some common options that can be used with CMake are:

* ``-DCMAKE_INSTALL_PREFIX=directory`` --- Specify for directory the full pathname of where you want the HPVM tools and libraries to be installed.
* ``-DCMAKE_BUILD_TYPE=type`` --- Valid options for type are Debug, Release, RelWithDebInfo, and MinSizeRel. Default is Debug.
* ``-DLLVM_ENABLE_ASSERTIONS=On`` --- Compile with assertion checks enabled (default is Yes for Debug builds, No for all other build types).

Now, compile the HPVM Compilation Tool ``hpvm-clang`` using:

.. code-block:: shell

   make -j<number of threads> hpvm-clang

With all the aforementioned steps, HPVM should be built, installed, tested and ready to use.
In particular, ``hpvm-clang`` should be an executable command from your command line.

Tests and Benchmarks
--------------------

We provide a number of general benchmarks, DNN benchmarks, and test cases, written in HPVM.

``make`` targets ``check-hpvm-pass``, ``check-hpvm-dnn``, ``check-hpvm-profiler``,
and others tests various components of HPVM.
You can run tests similarly as how ``hpvm-clang`` is compiled: for example,

.. code-block:: shell

   make -j<number of threads> check-hpvm-pass

runs ``check-hpvm-pass`` tests. See :doc:`/components/tests` for details on benchmarks and test cases.
