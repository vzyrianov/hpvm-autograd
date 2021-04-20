
Approximation Configuration Format
==================================

The HPVM binaries generated from the (Keras and PyTorch) Frontends support loading in a configuration file (`HPVM_binary -c ${config_file_path}`) that loads approximation knobs corresponding to each tensor operation in the program. This configuration file is the output of the autotuner (`predtuner`) that selects an approximation knob for each tensor operation, while respecting the accuracy degradation budget given for autotuning. The HPVM tensor runtime uses the configuration to dispatch to the corresponding approximate variants with appropriate arguments.


The format of the configuration is includes one line per fused HPVM node. Note that
this often includes multiple Tensor operations in a single fused node. For instance,
a Convolution, Add, and Relu, are fused into a single HPVM node since these are
semantically a convolution layer. This fusion is done to facilitate code
generation to accelerators and libraries that expose higher level abstractions
such as "Convolution Layers" or "Dense Layer" as the API.

File Format
--------------

.. code-block:: text
		
   +++++
   ${config_id} ${predicted_speedup} $predicted_energy} ${real_accuracy} ${accuracy_degration}
   ${hpvm_node_id} ${device=cpu|gpu} ${tensor_op_type} ${approximation_knob} ....
   ${hpvm_node_id} .....
   -----

The delimeters `+++++` and `-----` marked beginning and end of a configuration

The `$config_id` is the configuration ID in the configuration file. A configuration file is a list of multiple configurations - the runtime can select from any of these at runtime - default behavior is to use the first configuration in the file.

`$predicted_speedup` is the "hardware-agnostic" speedup predicted by the autotuner using a performance heursitic.

`$predicted_energy`: hardware-agnostic predicted energy metric. Currently, the tuner sets this to 0 - since we do not yet support energy estimation.

`$real_accuracy` is the accuracy of the program on the tune set (inputs used for tuning) when no approximations are applied and `$accuracy_degradation` is the drop in accuracy when applying the configuration (i.e., the specific knob settings).

`$hpvm_node_id` specifies the node ID to apply the approximation knobs for, `$device` specifies the device to offload to (GPU or CPU), `$tensor_op_type` specifies the type of tensor operation (conv, mul, add, relu etc.), and `$approximation_knob` is the knob setting corresponding to this tensor operation. The autotuner selects these knobs.

Approximation Knobs
--------------------

The `$approximation_knob` is an integer ID that represents an approximation knob.
HPVM currently supports `fp16` (knob value `12`) and `fp32` (knob value `11`) for all
types of tensor operations. For convolution operations, "sampling" and "perforation" are the
supported algorithmic approximations with the following knobs.


Perforation Knobs
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Knob
     - FP Precision
     - PerforationRate (%)
     - PerforationType
     - StartOffset
   * - 121
     - FP32
     - 50%
     - Col
     - 0
   * - 122
     - FP32
     - 50%
     - Col
     - 1
   * - 123
     - FP32
     - 50%
     - Row
     - 0
   * - 124
     - FP32
     - 50%
     - Row
     - 1
   * - 125
     - FP32
     - 33%
     - Col
     - 0
   * - 126
     - FP32
     - 33%
     - Col
     - 1
   * - 127
     - FP32
     - 33%
     - Col
     - 2
   * - 128
     - FP32
     - 33%
     - Row
     - 0
   * - 129
     - FP32
     - 33%
     - Row
     - 1
   * - 130
     - FP32
     - 33%
     - Row
     - 2
   * - 131
     - FP32
     - 25%
     - Col
     - 0
   * - 132
     - FP32
     - 25%
     - Col
     - 1
   * - 133
     - FP32
     - 25%
     - Col
     - 2
   * - 134
     - FP32
     - 25%
     - Col
     - 3
   * - 135
     - FP32
     - 25%
     - Row
     - 0
   * - 136
     - FP32
     - 25%
     - Row
     - 1
   * - 137
     - FP32
     - 25%
     - Row
     - 2
   * - 138
     - FP32
     - 25%
     - Row
     - 3

Note that knobs `151-168` have the exact same arguments in this order but use FP16 precision (available on GPUs).

Sampling Knobs
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Knob
     - FP Precision
     - SampingRate (%)
     - StartOffset
   * - 231
     - FP32
     - 50%
     - 0
   * - 232
     - FP32
     - 50%
     - 1
   * - 233
     - FP32
     - 33%
     - 0
   * - 234
     - FP32
     - 33%
     - 1
   * - 235
     - FP32
     - 33%
     - 2
   * - 236
     - FP32
     - 25%
     - 0
   * - 237
     - FP32
     - 25%
     - 1
   * - 238
     - FP32
     - 25%
     - 2
   * - 239
     - FP32
     - 25%
     - 3
  

Note that knobs `261-269` have the exact same arguments in this order but use FP16 precision (available on GPUs).



The mappings for these approximation knobs are parsed by  the HPVM Tensor Runtime from `hpvm/hpvm/projects/hpvm-tensor-rt/global_knobs.txt`.



