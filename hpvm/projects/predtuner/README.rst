Autotuning and Predictive Autotuning
====================================

PredTuner performs autotuning on program approximation knobs using an error-predictive proxy
in place of the original program, to greatly speedup autotuning while getting results
comparable in quality. ``current_version == 0.3``.

Read our `documentation here <https://predtuner.readthedocs.io/en/latest/index.html>`_
for how to install and use PredTuner.

Tuning with HPVM Binary
-----------------------

This branch (`hpvm`) contains beta support for HPVM binaries.
Please refer to `examples/tune_hpvm_bin.py` for an example with explanations.
