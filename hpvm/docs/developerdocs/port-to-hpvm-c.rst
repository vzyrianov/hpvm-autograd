Porting a Program from C to HPVM-C
==================================

The following represents the required steps to port a regular C program into an HPVM program with HPVM-C. These steps are described at a high level; for more detail, please see `hpvm-cava </hpvm/test/benchmarks/hpvm-cava>`_ provided in `benchmarks </hpvm/test/benchmarks>`_.

* Separate the computation that will become a kernel into its own (leaf node) function and add the attributes and target hint.
* Create a level 1 wrapper node function that will describe the thread-level parallelism (for the GPU). The node will:

  * Use the ``createNode[ND]()`` method to create a kernel node and specify how many threads will execute it.
  * Bind its arguments to the kernel arguments.

* If desired, create a level 2 wrapper node function which will describe the threadblock-level parallalism (for the GPU). This node will:

  * Use the ``createNode[ND]()`` method to create a level 1 wrapper node and specify how many threadblocks will execute it.
  * Bind its arguments to its child node's arguments.

* A root node function that creates all the top-level wrapper nodes, binds their arguments, and connects their edges.

  * Each root node represents a DFG.

* All the above node functions have the combined arguments of all the kernels that are nested at each level. 
* The host code will have to include the following:

  * Initialize the HPVM runtime using the ``init()`` method.
  * Create an argument struct for each DFG and assign its member variables.
  * Add all the memory that is required by the kernel into the memory tracker.
  * Launch the DFG by calling the ``launch()`` method on the root node function, and passing the corresponding argument struct.
  * Wait for the DFG to complete execution.
  * Read out any generated memory using the ``request_mem()`` method.
  * Remove all the tracked memory from the memory tracker.
