Frequently Asked Questions
==========================

#. **Is Python3.6 a strict requirement for installation?**

   Yes, our HPVM python packages require python version = 3.6.
   If you don't have a Python3.6 on your system, we encourage using the provided ``env.yaml`` conda environment.

#. **What is a "target device" or the "profiling stage"?
   Why does the tutorial seems to suggest building HPVM** :ref:`on a second device<target-profiling>`?

   HPVM is capable of *predictive approximation tuning* which, due to its computational cost,
   is often done on a powerful computer, like a server,
   but the selected approximations are usually used to speedup your application
   on a less powerful device (the *target device*, such as an edge device).
   The profiling stage (using `hpvm-profiler`) is necessary so that the real speedup of approximations are measured,
   and this is also done on the target device.
   See our `ApproxTuner paper <https://dl.acm.org/doi/10.1145/3437801.3446108>`_ for more details on this.

   Currently, HPVM must be built on both the server and the target device for this purpose.
   We will achieve better server/edge separation of HPVM in the following releases,
   so that only the necessary part of code are built on each device.

#. **What is the expected speedup with approximations on my target device?**

   The approximation implementations in HPVM are currently only optimized for
   `Nvidia Tegra TX2 <https://developer.nvidia.com/embedded/jetson-tx2>`_.
   The routines may not provide the same speedup on other hardware devices --
   though systems with similar hardware specifications may exhibit similar performance.
   We are working on providing speedups across a wider range of devices.

#. **Why doesn't the conda environment / Python packages installation work on Jetson boards?**

   You may be seeing errors like 

   .. code-block:: text

      ResolvePackageNotFound:
        pytorch==1.6.0

   or other errors indicating ``pytorch``, ``torchvision`` or other packages cannot be installed,
   because these packages are not prebuilt for ARM CPU on `PyPI <https://pypi.org/>`_.

   The simplest solution is not to install HPVM frontends and autotuner;
   see :ref:`this <skip-pypkg>` for how to do so.
   The job of these packages are best left to a server machine.

#. **What to do when running into "CUDA out of memory" errors?**

   When the Keras/PyTorch frontends generates code, they accept a "batch size" parameter,
   which decides the batch size at which the DNN inference runs.
   You may need to reduce batch size when encountering out of memory errors.

#. **How many autotuning iterations should I use with PredTuner package in HPVM?**

   The number of tuning iterations required to achieve good results varies across benchmarks
   and should be figured out on a per-benchmark basis.
   For the included 10 CNNs, we recommmend using at least 10K iterations.

#. **Does this release support combining HPVM tensor and non-tensor operations in a single program?**

   Currently we do not support tensor and non-tensor code in the same application.
   We will support this feature in the next release.

#. **Does this release support object detection models?** 

   Currrently, HPVM doesn't support object detection models,
   due to the limited number of operators supported in the tensor library `hpvm-tensor-rt`.
   We will add support for more operators in the next release.
