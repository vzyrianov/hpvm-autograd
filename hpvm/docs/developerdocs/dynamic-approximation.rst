Dynamic Approximation Control
---------------------------------------

HPVM includes support for dynamically updating the approximation configuration used for processing an input Batch.
The Keras and PyTorch frontends generate batched loops (each processes a batch of inputs)
that load and apply a particular configuration.
In the main batched loop the frontends generate a call to `llvm_hpvm_invokeRtControl`
which invokes the dynamic approximation control.
In the default mode, the runtime controller does not switch configuration -- it loads the first configuration in the configuration file.
The configuration file is described here in :doc:`configuration-format`.
The configurations loaded from the configuration file are organized into a Pareto curve
that is used for dynamic tuning.
The code for the runtime controller exists under
``projects/hpvm-tensor-rt/tensor_runtime/src/hpvm-rt-controller.cpp``.

The runtime can be configured to switch configurations to respond to system slowdowns
(e.g., caused by reduced frequency). 
This release does not include a demo that shows the dynamic approximation capability in HPVM
but users can use/repurpose the code to cater to custom use cases.
To use this mode, users must do the following steps:

* The first line in the configuration line must include the target batch processing time (in milliseconds).
  Users can use this to specify any (soft) time constraints for processing a single batch of inputs.
  The HPVM runtime controller uses this target time and the measured time to compute the required speedup,
  and uses it to select a configuration that provides the required speedup. 

* Modify ``projects/hpvm-tensor-rt/tensor_runtime/src/hpvm-rt-controller.cpp`` to update the 
  `llvm_hpvm_invokeRtControl` macro at the top of the file. Uncomment this line: 

.. code-block:: C++

    #define llvm_hpvm_invokeRtControl_ADJUST_PR llvm_hpvm_invokeRtControl

This enables the probabilistic configuration selection mode in HPVM.
The probalistic selection is needed since often no configuration on the Pareto curve offers the exact required speedup.
The probabilistic selection mode probabilistically picks among 2 configurations so as to provide the target speedup over time.
