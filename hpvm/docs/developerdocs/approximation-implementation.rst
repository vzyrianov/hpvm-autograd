Approximate Algorithm Implementations
=========================================

This release includes two approximations, namely perforation and sampling, both of which apply to convolution operations. The knobs for these approximations are described in :doc:`configuration-format`. We include implementations for these approximations on both GPU and CPU. The implementations are described below: 

Perforated Convolutions
-----------------------

Overview
^^^^^^^^^

The core idea of perforated convolutions is to compute a subset of the output tensor elements and interpolate the missing elements. Specifically, we include an implementation that skips entire output rows or columns (configurable through knobs), and interpolates the missing rows/columns through neighbour averaging.
Since the approximation reduces the number of output elements that need to be computed, it reduces the multiply-accumulate (MAC) operations, and reduces memory bandwidth usage (loads subset of data), hence resulting in both speedups and energy reductions. Our implementation performs the perforation at fixed strides (e.g. skip 1 out of every 3 rows) and the rate of perforation is a configurable knob. The type of perforation (row/col) is also configurable. Another knob is the starting offset - the tensor row/column index to start the perforation (i.e., skipping) from. 

Description
^^^^^^^^^^^

Our implementation for perforated convolution is a three-step process:

* **Patch matrix creation:** Based on indices of the rows/columns to be perforated, the corresponding elements of the input tensor are used to create a new matrix called an input-patch matrix. The input-patch matrix is a matrix laid out in memory such that convolution is reduced to a simple matrix multiplication operation. This approach is similar to one described in this `paper <https://dl.acm.org/doi/abs/10.1145/2964284.2967243>`__.

* **Dense matrix multiplication:** This step involves performing a matrix multiplication in a manner very similar to described in this `paper <https://arxiv.org/pdf/1704.04428.pdf>`__. Note that these matrices are dense (no sparsity in the tensor representation).

* **Interpolation of missing values:** A new tensor is created with dimensions the output tensor is expected to have. The perforated tensor output (after matrix multiplication) is copied to corresponding rows/columns, and for the skipped rows/columns, the output values are populated using interpolation; using arithmetic mean of the neighboring elements. For column perforation, these neighboring elements are the right and left element of the skipped element. For row perforation, the top and bottom neigbouring elements are used for interpolation. The output of this step is the approximate (perforated) convolution result.

Filter Sampling
---------------

Overview
^^^^^^^^^
The core idea of filter sampling is to use a subset of convolution filter elements and inputs to compute the (full) tensor output. Filter sampling is a variant of "input sampling", while perforation is a variant of "output sampling". Similar to perforation, filter sampling also reduces MAC operations and memory bandwidth usage. The filter elements (and corresponding input tensor elements) are skipped at a regular stride. The start offset is a tunable knob (exposed to our autotuner) which controls the filter tensor index at which skipping starts. 

Description
^^^^^^^^^^^

Our filter sampling implementation involves the following steps:

* **Creation of sampled filter:** This step creates a new sampled filter (with fewer elements) whose size is based on the sampling rate (and offset). Since filter elements are skipped, the remaining elements are scaled up by the factor of  rate / (rate - 1) and copied to the newly allocated sampled fiter. Scaling up of filter elements helps make up for the lost accuracy from sampling the filter (found as part of our empirical study).

* **Patch matrix creation:** Based on filter element indices used in the construction of the sampled filter, the corresponding input tensor elements are used to create a new matrix, called an input-patch matrix. The input-patch matrix is a matrix laid out in memory in such a way that convolution is transformed to a simple matrix multiplication operation. 

* **Dense matrix multiplication:** This step involves performing a matrix multiplication on the (sampled) filter and input patch matrices. The output result of matrix multiplication is the approximate convolution result. 


Sources 
^^^^^^^^

The implementation for perforation and sampling for GPU are present in `hpmv/hpvm/projects/hpvm-tensor-rt/tensor_runtime/src/approx_techniques.cu`

Relevant Routines 
 * `tensorConvApprox` (FP32)
 * `tensorConvApproxHalf2` (FP16)

The implementations on CPU are present in: `projects/hpvm-tensor-rt/tensor_runtime/src/tensor_cpu_runtime.cc`. The Relevant routine is: `tensorConvApproxCPU`. Note that this single routine supports baseline (no approximation), perforation, and sampling knobs. All supported knobs are detailed in :doc:`configuration-format`.



