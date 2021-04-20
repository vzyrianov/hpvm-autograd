.. _contents:

PredTuner
=============================

PredTuner is a Python library for predictive approximation autotuning.

PredTuner performs autotuning on approximation choices for a program
using an error-predictive proxy instead of executing the program,
to greatly speedup autotuning while getting results of comparable quality.

PredTuner is a main component of `ApproxTuner
<https://ppopp21.sigplan.org/details/PPoPP-2021-main-conference/41/ApproxTuner-A-Compiler-and-Runtime-System-for-Adaptive-Approximations>`_.


Solution for Efficient Approximation Autotuning
-----------------------------------------------

- Start a tuning session in 10 lines of code
- Deep integration with PyTorch for DNN supports
- Multiple levels of APIs for generality and ease-of-use
- Effective accuracy prediction models
- Easily store and visualize tuning results in many formats

Documentation
-------------

.. toctree::
   :maxdepth: 1

   getting_started
   reference/index

Indices and tables
------------------

* :ref:`genindex`
