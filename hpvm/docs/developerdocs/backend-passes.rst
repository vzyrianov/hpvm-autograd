HPVM Backend Passes
====================

HPVM includes multiple backend targets for code generation. The transformations to target these devices utilise a common utility class `CodeGenTraversal`, where each node in the HPVM Dataflow graph in a bottom up  topological ordering. For many target backends, some different logic is required when processing Internal Nodes from Leaf Nodes. As such the `CodeGenTraversal` utility class provides 2 virtual functions, namely `codeGen(DFInternalNode* )` and `codeGen(DFLeafNode* )` to generate target specific LLVM IR for each node function as well as adding HPVM runtime calls where needed.  

DFG2LLVM_CPU
^^^^^^^^^^^^
Description
-----------

The CPU backend plays an important role in HPVM codegen. It enables targeting devices that only host a CPU. Additionally, it is used in the compilation pipeline involving other backends. For instance, it is used for inserting runtime calls for data movement and runtime calls for launching compute kernels. 

For Leaf Nodes targeting the CPU, this backend generates sequential code that uses loops to represent the dynamic replication of those nodes. We are currently working on an extension to this backend where multi-dimensional nodes can be executed in parallel.

codeGen(DFLeafNode* )
----------------------
Consider a 3 dimensional Leaf node function with the following body:

.. code-block:: c

  void  leaf1(int* p, size_t pSize, int* q, size_t qSize){

  void* mynode = __hpvm__getNode( );
  size_t i = __hpvm__getNodeInstanceID_x(mynode); 
  size_t = __hpvm__getNodeInstanceID_y(mynode);
  size_t k = __hpvm__getNodeInstanceID_z(mynode);

  size_t xDim = __hpvm__getNumNodeInstances_x(mynode); 
  size_t yDim = __hpvm__getNumNodeInstances_y(mynode);
  size_t zDim = __hpvm__getNumNodeInstances_z(mynode);

  size_t index = i * (yDim + zDim) + j * zDim + k;

  p[index] = p[index] + q[index];

  __hpvm_return(4, p, pSize, q, qSize); 

  }

The above example will illustrate the steps in transforming the leaf node in this pass. 


1. A clone of the function is made, with 6 additional arguments added to its argument list which correspond to the current dimension indices and dimension sizes along the x,y,z axes. Any node querying intrinsics on the current node will be updated to refer to these newly added arguments instead, and these calls will be removed.

.. code-block:: c

  void  leaf1_clone(int* p, size_t pSize, int* q, size_t qSize,
  size_t idx_x, size_t idx_y, size_t idx_z,
  size_t dim_x, size_t dim_y, size_t dim_z){

    void* mynode = __hpvm__getNode( );
    size_t i = idx_x;  // replaced  __hpvm__getNodeInstanceID_x(mynode); 
    size_t j = idx_y;  // replaced  __hpvm__getNodeInstanceID_y(mynode); 
    size_t k = idx_z;  // replaced  __hpvm__getNodeInstanceID_z(mynode); 

    size_t xDim = dim_x; // replaced __hpvm__getNumNodeInstances_x(mynode); 
    size_t yDim = dim_y; // replaced __hpvm__getNumNodeInstances_y(mynode);
    size_t zDim = dim_z; // replaced __hpvm__getNumNodeInstances_z(mynode);


    size_t index = i * (yDim + zDim) + j * zDim + k;

    p[index] = p[index] + q[index];

    __hpvm_return(4, p, pSize, q, qSize); 

  }

2. An additional BasicBlock is inserted at the entry to the function and calls to the hpvm runtime for each pointer argument are inserted.

.. code-block:: c

  void  leaf1__clone(int* p, size_t pSize, int* q, size_t qSize,
  size_t idx_x, size_t idx_y, size_t idx_z,
  size_t dim_x, size_t dim_y, size_t dim_z){

    llvm_hpvm_cpu_argument_ptr( (void*) p , pSize);
    llvm_hpvm_cpu_argument_ptr( (void*) q , qSize);

    size_t i = idx_x;
    size_t j = idx_y;
    size_t k = idx_z;

    size_t xDim = dim_x; 
    size_t yDim = dim_y;
    size_t zDim = dim_z;

    size_t index = i * (yDim + zDim) + j * zDim + k;

    p[index] = p[index] + q[index];

    __hpvm_return(4, p, pSize, q, qSize); 

  }

3. Later on, when the parent (internal) node for this leaf node will process, it will create nested for-loops over each of the dimensions. Inside the loop body, the leaf node function is called with appropriate arguments.

If the original call was:

`void* leaf =  __hpvm__createNodeND(3, (void*) leaf1_clone  , 5 , 6 , 7);`

The code would be transformed to:

.. code-block:: c

  for(int i = 0; i < 5; i++){
    for(int j=0; j < 6; j++){
      for(int k = 0; k < 7; k++){
        leaf1_clone(p, pSize, i, j, k, 5, 6, 7);
      }
    }
  }

codeGen(DFInternalNode* )
-------------------------

For Internal nodes in the CPU backends, the CPU version of that node is only generated if all its children nodes have CPU as it’s target device. 

1. First, it checks if the immediate child nodes of the current internal node are also targeted for CPU.

2. If yes, it creates a cloned version of the internal node function. This node function will have 6 additional arguments added to it, similar to leaf node case.

3. As in the leaf node case, the internal node is responsible for converting each multi-dimensional child nodes into for loops along each axis.

4. Each Internal Node has an entry node for the Child Subgraph as well as an exit node for the child subgraph. The exit node return values are appropriately updated to reflect the newly cloned nodes.

Depending on whether the rootNode’s subgraph is streaming or non-streaming, the generated launch function behavior is varied. 

* For non-streaming launches:

1. A wrapper root node function is created which takes an opaque pointer to the packed struct and returns an i8*. 

2. Inside the wrapper root node function, the actual root node function is invoked by extracting the correct arguments from the input struct.

3. Similarly the output struct (if the node returns) is the last element of the input struct and appropriate store instructions are generated. The uses of the original launch function are also updated to use the new wrapper root function.

* For streaming launches:

1. Similar to the non-streaming case, a wrapper root node function is created which takes the packed struct of arguments and the graph handle identifier.

2. For each incoming argument to that function, it checks if that edge is streaming. If yes, it extracts the value from the struct and creates a buffer for it in the newly created function.

3. For each type of edge in the child graph of the root node function, it creates appropriate bindIn_buffer, bindOut_buffer, edge_buffer calls based on the extracted buffers. 

4. An additional buffer is created which is used to indicate whether the following buffer input is the last input.

5. A function filter is created for the root node function. This filter function essentially reads the arguments from the input buffer, applies the root node function on these arguments, and pushes the results onto the output buffers.

6. Additionally this function creates a loop around the invocation checking each time if the isLastInput buffer is set to true.


DFG2LLVM_CUDNN
^^^^^^^^^^^^^^
Description
-----------

This backend targets calls in `hpvm-tensor-rt` that in turn call tensor library calls in the cuDNN library.
This backend converts (in most cases one-to-one) tensor intrinsic calls to corresponding runtime calls.

codeGen(DFLeafNode* )
----------------------

Consider the following leaf node function which performs a tensor convolution:

.. code-block:: c

  void conv_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {

  __hpvm__hint(hpvm::TENSOR_TARGET);

  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);

  __hpvm__return(2, r, (size_t)0);
  }

1. A clone of the function is made and appended with ‘_cudnn’ in its name. Additionally the nounwind attribute is added to the cloned function.

2. For each of the pointer type arguments for the node function, a hpvm-tensor-runtime call requesting the tensor to be copied to the GPU target device. The assumption is that each pointer argument is a tensor type. 


.. code-block:: c

  void conv_node_cudnn(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {

  __hpvm_request_tensor(t1, /* GPU */ 1);
  __hpvm_request_tensor(t2, /* GPU */ 1);

  __hpvm__hint(hpvm::TENSOR_TARGET);

  __hpvm__attributes(2, t1, t2, 0);

  void *r = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);

  __hpvm__return(2, r, (size_t)0);
  }


3. The rest of the pass can be viewed as a dictionary mapping from HPVM intrinsics representing tensor operations such as convolutions to their corresponding CuDNN functions.

.. code-block:: c

  void conv_node_cudnn(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {

  __hpvm_request_tensor(t1, /* GPU */ 1);
  __hpvm_request_tensor(t2, /* GPU */ 1);

  __hpvm__hint(hpvm::TENSOR_TARGET);

  __hpvm__attributes(2, t1, t2, 0);

  void *r = tensorConvolution(t1, t2, 2, 2, 4, 4, /* CuDNN conv mode */ 1, /* CuDnn conv Precision */ 0);

  __hpvm__return(2, r, (size_t)0);
  }

4. Most definitions of the intrinsics arguments map almost identically to their CUDNN implementations. For some intrinsics, these are mapped to a single runtime call with different function arguments. For example, max pooling and mean pooling are separate intrinsics in HPVM, but they both get mapped to tensorPooling in CUDNN, with an integer specifying the type of pooling. 

5. The tensor runtime (`hpvm-tensor-rt`) and HPVM runtime (`hpvm-rt`) are currently not integrated, and as such this pass inserts the `initTensorRuntime` calls (for `hpvm-tensor-rt` initialization) before the `hpvm-rt` init call. Similarly, it inserts the tensor runtimes cleanup call before the HPVM runtime cleanup call. (Assertion that both runtime calls can only be used once in the entire module).

codeGen(DFInternalNode* )
-------------------------
Internal Nodes are skipped in this backend pass.


FuseHPVMTensorNodes
^^^^^^^^^^^^^^^^^^^
Description
-----------

For users writing tensor code though our frontends (e.g. Keras, C++, PyTorch), each tensor operation is mapped to its own HPVM Dataflow (Leaf) Node, with appropriate HPVM edge bindings feeding the output of one layer into the next. The FuseHPVMTensorNodes pass combines specific patterns of tensor operations from multiple separate nodes into a single HPVM leaf node. 

codeGen(DFLeafNode* )
---------------------

While the pass is generic, we only support `TENSOR_TARGET` (this hint implies HPVM nodes with tensor operations) nodes for fusion. 
Additionally each leaf node is first identified as being a valid HPVM tensor node (i.e. contains HPVM intrinsics as the first intrinsic). 

Consider the following consecutive leaf nodes:

.. code-block:: c

  void conv_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    __hpvm__return(2, r, (size_t)0);
  }

  void add_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_add(t1, t2);
    __hpvm__return(2, r, (size_t)0);
  }

  void relu_node(void *t1, size_t bytes_t1) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_relu(t1);
    __hpvm__return(2, r, (size_t)0);
  }

  void pool_max_node(void *t1, size_t bytes_t1) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_pool_max(t1, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r, (size_t)0);
  }


1. Originally, each tensor operation is mapped to it’s unique HPVM node. This pass identifies the sequence of operations across consecutive nodes. If this sequence matches a fusion pattern, then all those operations are copied into one newly created node.

The exhaustive list of patterns which are fused are:

* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation} -> llvm.hpvm.tensor.{pooling}
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation} 
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{pooling}
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add
* llvm.hpvm.tensor.mul -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation}
* llvm.hpvm.tensor.mul -> llvm.hpvm.tensor.add


According to the list above, the nodes satisfy the pattern 1.

2. It checks if each node in the pattern belongs to the same target. Note that all nodes above are labelled as `TENSOR_TARGET`.

3. The pass collects these node handles into a fusion target as:

* `conv_node -> add_node -> relu_node -> pool_max_node`


4. Once the pass has collected the list of all fusion targets (sets of HPVM nodes to fuse), it fuses these iteratively. 

5. Each pair of nodes is fused together into a single node and then reinserted into the beginning of the fusion target list. For example, first the `conv_node` and `add_node` will be fused creating `fused_node_1` and then the state of the list will be:

* `fused_node_1 -> relu_node -> pool_max_node`

.. code-block:: c

  void fused_node_1(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2,
                                  void *t3, size_t bytes_t3) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    __hpvm__return(2, r2, (size_t)0);
  }


  void relu_node(void *t1, size_t bytes_t1) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_relu(t1);
    __hpvm__return(2, r, (size_t)0);
  }

  void pool_max_node(void *t1, size_t bytes_t1) {
    __hpvm__hint(hpvm::TENSOR_TARGET);


    void *r = __hpvm__tensor_pool_max(t1, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r, (size_t)0);
  } 


6. In the next step, `fused_node_1` and `relu_node` be fused together. These steps are repeated until only a single node remains in the fusion target list (i.e. until all nodes are fused into a single node). 

.. code-block:: c

  void all_fused(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3, size_t bytes_t3) {
    __hpvm__hint(hpvm::TENSOR_TARGET);

    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    void *r3 = __hpvm__tensor_relu(r2);
    void *r4 = __hpvm__tensor_pool_max(r3, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r4, (size_t)0);
  }


7. During pairwise fusion, the entire argument list of the first node function is included. For the second node function, only those arguments which are not bound to the DFG edges between these two nodes are appended into the argument lists. 

8. As we can see in our example above, as relu and pooling operate on the same input tensor, no additional argument was included. However for addition after convolution, another tensor argument is created to account for that.


9. Similarly, these DFG edges are replaced with SSA value uses in this new function. A new `createNodeND` call is created for this newly fused function and then the edges and parent node are updated accordingly to maintain semantics. This step is then again repeated, creating new nodes and updating edges accordingly. 



codeGen(DFInternalNode* )
-------------------------
Internal Nodes are skipped in this backend pass.

DFG2LLVM_WrapperAPI
^^^^^^^^^^^^^^^^^^^
Description
-----------

This pass is responsible for "pattern matching" multiple tensor operations inside HPVM
nodes so that the appropriate set of operations are replaced with a single
call to a runtime routine. This allows the HPVM IR to represent a graph
with tensor operations in a target-agnostic manner.

Let’s consider the end result of the `FuseHPVMTensorNodes` example:

.. code-block:: c

  void all_fused(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3, size_t bytes_t3) {


    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    void *r3 = __hpvm__tensor_relu(r2);
    void *r4 = __hpvm__tensor_pool_max(r3, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r4, (size_t)0);
  }

Similar to the FuseHPVMTensorNodes example, the DFG2LLVM_WrapperAPI pass also has fusion patterns. However in this pass, the tensor operations are within a single node.

The exhaustive list of patterns is shown below:

* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation} -> llvm.hpvm.tensor.{pooling}
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation}
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{pooling}
* llvm.hpvm.tensor.convolution -> llvm.hpvm.tensor.add
* llvm.hpvm.tensor.mul -> llvm.hpvm.tensor.add -> llvm.hpvm.tensor.{activation}
* llvm.hpvm.tensor.mul -> llvm.hpvm.tensor.add


codeGen(DFLeafNode* )
---------------------

1. Our example above maps to the first fusion pattern. First a clone of the function is made and hpvm-tensor-runtime calls are added to the beginning of the clone for each tensor argument.

.. code-block:: c

  void all_fused_wrapper_api(void *t1, size_t bytes_t1, void *t2,size_t bytes_t2, void *t3, size_t bytes_t3) {
    __hpvm_request_tensor(t1, /* GPU */ 1);
    __hpvm_request_tensor(t2, /* GPU */ 1);
    __hpvm_request_tensor(t3, /* GPU */ 1);




    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    void *r3 = __hpvm__tensor_relu(r2);
    void *r4 = __hpvm__tensor_pool_max(r3, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r4, (size_t)0);
  }


2. For each of the patterns listed previously, a specific ‘wrapper’ function exists which when invoked by the runtime carries out all of the operations. For the pattern above (convolution layer) the corresponding wrapper call is `wrapper_ConvLayer2`. The first argument to these wrapper functions is the ID of the HPVM node - this is added by the frontend to assign a linear ordering to HPVM nodes.

.. code-block:: c

  void all_fused_wrapper_api(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3, size_t bytes_t3) {
    __hpvm_request_tensor(t1, /* GPU */ 1);
    __hpvm_request_tensor(t2, /* GPU */ 1);
    __hpvm_request_tensor(t3, /* GPU */ 1);


    void* w1 = wrapper_ConvLayer2("all_fused_wra..." /* , ... */);  // some arguments omitted


    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    void *r3 = __hpvm__tensor_relu(r2);
    void *r4 = __hpvm__tensor_pool_max(r3, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r4, (size_t)0);
  }

3. The remaining arguments of the wrapper_convLayer2 call are taken from the arguments passed to the individual tensor operations from which the fused call is made.

.. code-block:: c

  void all_fused_wrapper_api(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2, void *t3, size_t bytes_t3) {
    __hpvm_request_tensor(t1, /* GPU */ 1);
    __hpvm_request_tensor(t2, /* GPU */ 1);
    __hpvm_request_tensor(t3, /* GPU */ 1);


    void* w1 = wrapper_ConvLayer2("all_fused_wra...", t1, t2, t3, 2, 2, 4, 4, 0, 3, 3, 0, 0, 2, 2, 0.0, 0.0);

    void *r1 = __hpvm__tensor_convolution(t1, t2, 2, 2, 4, 4);
    void *r2 = __hpvm__tensor_add(r1, t3);
    void *r3 = __hpvm__tensor_relu(r2);
    void *r4 = __hpvm__tensor_pool_max(r3, 3, 3, 0, 0, 2, 2);
    __hpvm__return(2, r4, (size_t)0);
  }

4. Finally, the original operations are removed and the final values uses are replaced with the wrapper function call result.

codeGen(DFInternalNode* )
-------------------------
Internal Nodes are skipped in this backend pass.

DFG2LLVM_OpenCL
^^^^^^^^^^^^^^^^^^^
Description
-----------
This backend generates GPU kernel code and code for 
launching kernels for the GPU target using HPVM dataflow graph. The kernels are
generated into a separate file which is the C-Backend uses to generate 
OpenCL kernels with.

The pass begins by first creating a clone of the entire module, which will be used to generate the kernel code, and subsequently removing all global variables, functions, and global aliases from that clone. That cloned module’s Datalayout and Target Triple is updated to target the OpenCL GPU backend.


codeGen(DFLeafNode* )
^^^^^^^^^^^^^^^^^^^^^
1. The Leaf node queries it’s parent’s hierarchical structure to identify which node would be responsible for launching the GPU Kernel.

2. If it’s parent node does not have any replication factor across the dimensions, it would be considered the launching node.

3. Otherwise, we assume that parent node to represent GPU thread blocks and go up another level in the hierarchy to get the Kernel launching node.

4. The Kernel node is cloned and inserted into the new Kernel Module file created which has target information for the GPU. It’s function attributes are removed and the nounwind attribute is added to it.

5. We then iterate over all incoming data flow edges into this leaf node and if the source node of that edge is not the entry node, it is assigned the allocation node of the Kernel. The Allocation node implies there is shared memory that will be accessed by the different threads of the leaf node.

6. The Allocation Node Function is also cloned, and is iterated over to identify hpvm_malloc calls. These hpvm_malloc calls are removed and their uses are replaced with null. 

7. None shared memory pointer arguments of the function are moved to the global address space of the kernel. Additionally, those global memory arguments whose load accesses are independent of the current node id are moved to constant memory as an optimization.

8. Finally, hpvm intrinsics inside the leaf node for querying the graph structure (such as  `__hpvm__getNodeInstanceID_*` and `__hpvm__getNumNodeInstances_*`) are replaced with GPU specific function calls such as  `get_global_id()`, `get_local_id()` or `get_group_id()`.

codeGen(DFInternalNode* )
^^^^^^^^^^^^^^^^^^^^^^^^^
1. For Internal Nodes whose leaf nodes would have been processed into OpenCL kernels, the Internal node would have been assigned as the Kernel launching node.

2. First an empty clone of the internal node function is created and a single basic block is inserted into it. The node function is appended with 6 additional arguments (just as with the CPU case).

3. A runtime call to `llvm_hpvm_ocl_launch`  with the kernel module file name and the specific kernel function is created by the hpvm_init. The graphID identifier is stored into a global variable at  the hpvm initialization time, and that handle is loaded into a variable inside the cloned internal node function.

4. The internal node is then instrumented with the hpvm runtime calls for specifying which variables to the kernel are input and their respective sizes. Similarly, whether they need to be copied back from the kernel, and the calls to perform the copying back. 

5. `llvm_hpvm_ocl_argument_shared` calls to already identified shared variables tell the runtime to mark specific memory as being shared.

6. Finally, the actual GPU execution parameters are calculated for factors such as the number of dimensions, as well as local and global work group sizes. Then calls to execute the kernel on the GPU and then wait for it’s execution are finally added.

