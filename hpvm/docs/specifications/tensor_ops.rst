Tensor Operations in HPVM
==========================

Tensor Implementation in HPVM
--------------------------------

Tensors are referred to in HPVM IR with pointers (as opposed to SSA tensor values). The HPVM Tensor Runtime allocates tensors in memory and is responsible for managing them. The runtime assumes that the layout of tensors is NCHW. 

Intrinsics for Tensor Operations
--------------------------------

Tensor Add
^^^^^^^^^^^

**Overview**

Add tensor pointed to by input with the tensor pointed to by bias. Axis of size 1 will be broadcast. Rank of input and bias tensors must match. Return a pointer to the resultant tensor of the same size and shape.

``i8* llvm.hpvm.tensor.add(i8* input , i8* bias)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - bias
     - Reference to the bias tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Multiplication
^^^^^^^^^^^^^^^^^^^^^

**Overview**

Perform matrix multiplication between the tensor pointed to by input and the tensor pointed to by weight. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.mul(i8* input, i8* weight)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - weight
     - Reference to the weight tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor ReLU
^^^^^^^^^^^^

**Overview**

Perform the element-wise clipped  ReLU (Rectified Linear Unit) operation on the tensor pointed to by input. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.relu(i8* input)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Clipped ReLU
^^^^^^^^^^^^^^^^^^^^

**Overview**

Perform the element-wise clipped ReLU (Rectified Linear Unit) operation on the tensor pointed to by input. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.clipped.relu(i8* input)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Tanh
^^^^^^^^^^^^

**Overview**

Perform the element-wise hyperbolic tangent operation on the tensor pointed to by input. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.tanh(i8* input)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Sigmoid
^^^^^^^^^^^^^^^

**Overview**

Perform the element-wise sigmoid function on the tensor pointed to by input. Sigmoid function: output = 1 / (1 + exp(-input). Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.sigmoid(i8* input)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Softmax
^^^^^^^^^^^^^^^

**Overview**

Perform the element-wise softmax operation on the tensor pointed to by input. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.softmax(i8* input)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Convolution
^^^^^^^^^^^^^^^^^^^

**Overview**

Perform convolution operation between the tensor pointed to by input and the tensor pointed to by filter. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.convolution(i8* input, i8* filter, i32 vpad, i32 hpad, i32 vstride, i32 hstride)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - filter
     - Reference to the filter tensor
   * - vpad
     - Vertical pad
   * - hpad
     - Horizontal pad
   * - vstride
     - Vertical stride
   * - hstride
     - Horizontal stride

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Group Convolution
^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

Perform depthwise-convolution operation between the tensor pointed to by input and the tensor pointed to by filter. This operation entails performing 2D convolutions separately over each channel of the given input and filter tensors. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.group.convolution(i8* input, i8* filter, i32 vpad, i32 hpad, i32 vstride, i32 hstride)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - filter
     - Reference to the filter tensor
   * - vpad
     - Vertical pad
   * - hpad
     - Horizontal pad
   * - vstride
     - Vertical stride
   * - hstride
     - Horizontal stride

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Batchnorm
^^^^^^^^^^^^^^^^^

**Overview**

Perform batch-normalization operation on the tensor pointed to by input. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.batchnorm(i8* input, i8* gamma, i8* beta, i8* mean, i8* variance, double epsilon)``

Batchnorm is computed using the following formula:
output = beta + gamma * ((input - mean)/sqrt(epsilon + variance))


**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - gamma
     - Reference to the scale tensor
   * - beta
     - Reference to the bias tensor
   * - mean
     - Reference to the mean tensor
   * - variance
     - Reference to the variance tensor
   * - epsilon
     - The Epsilon value in the batchnorm formula

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Pool Max
^^^^^^^^^^^^^^^^

**Overview**

Perform reduction maximum function to all elements within the sliding window, and place the maximum value in the output tensor. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.pool.max(i8* input, i8* filter, i32 winWidth, i32 winHeight, i32 vpad, i32 hpad, i32 vstride, i32 hstride)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - filter
     - Reference to the filter tensor
   * - winWidth
     - Width of the sliding window
   * - winHeight
     - Height of the sliding window
   * - vpad
     - Vertical pad
   * - hpad
     - Horizontal pad
   * - vstride
     - Vertical stride
   * - hstride
     - Horizontal stride

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Pool Min
^^^^^^^^^^^^^^^^

**Overview**

Perform reduction minimum function to all elements within the sliding window, and place the minimum value in the output tensor. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.pool.min(i8* input, i8* filter, i32 winWidth, i32 winHeight, i32 vpad, i32 hpad, i32 vstride, i32 hstride)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - filter
     - Reference to the filter tensor
   * - winWidth
     - Width of the sliding window
   * - winHeight
     - Height of the sliding window
   * - vpad
     - Vertical pad
   * - hpad
     - Horizontal pad
   * - vstride
     - Vertical stride
   * - hstride
     - Horizontal stride

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor


Tensor Pool Mean
^^^^^^^^^^^^^^^^^

**Overview**

Perform reduction mean function to all elements within the sliding window, and place the mean value in the output tensor. Return a pointer to the resultant tensor.

``i8* llvm.hpvm.tensor.pool.mean(i8* input, i8* filter, i32 winWidth, i32 winHeight, i32 vpad, i32 hpad, i32 vstride, i32 hstride)``

**Operands**

.. list-table::
   :header-rows: 1

   * - Operand
     - Descrption
   * - input
     - Reference to the input tensor
   * - filter
     - Reference to the filter tensor
   * - winWidth
     - Width of the sliding window
   * - winHeight
     - Height of the sliding window
   * - vpad
     - Vertical pad
   * - hpad
     - Horizontal pad
   * - vstride
     - Vertical stride
   * - hstride
     - Horizontal stride

**Result**

.. list-table::
   :header-rows: 1

   * - Result
     - Descrption
   * - output
     - Reference to the output tensor
