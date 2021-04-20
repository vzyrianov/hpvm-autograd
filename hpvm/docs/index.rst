.. _contents:

The HPVM Compiler Infrastructure
================================

HPVM is a compiler for heterogeneous parallel systems.
For more about what HPVM is, see `our website <https://publish.illinois.edu/hpvm-project/>`_
and publications:
`HPVM (PPoPP'18) <https://dl.acm.org/doi/pdf/10.1145/3200691.3178493>`_,
`ApproxHPVM (OOPSLA'19) <https://dl.acm.org/doi/10.1145/3360612>`_,
`ApproxTuner (PPoPP'21) <https://dl.acm.org/doi/10.1145/3437801.3446108>`_.

This is the documentation of HPVM at **version 1.0**.

Audience
--------

The intended audience for HPVM includes researchers and developers working in the areas of
compilers, programming languages, approximate computing, software optimization,
static and dynamic program analysis, and systems for machine learning.

`HPVM <https://dl.acm.org/doi/pdf/10.1145/3200691.3178493>`_
includes a retargetable compiler infrastructure that targets CPUs, GPUs, and accelerators
(this release does not include accelerator support)
and uses a portable compiler IR that explicitly represents data flow at the IR level,
It supports task, data, and pipelined parallelism
HPVM provides an extensible platform that compiler and programming languages
researchers can use as part of their work.

`ApproxHPVM <https://dl.acm.org/doi/10.1145/3360612>`_
and `ApproxTuner <https://dl.acm.org/doi/10.1145/3437801.3446108>`_
extend the HPVM compiler with support for high-level linear algebra tensor operations
(e.g., convolution, matrix multiplication)
and a framework that optimizes tensor-based programs using approximations
that tradeoff accuracy for performance and/or energy.
ApproxHPVM and ApproxTuner support many popular CNN models.
ApproxTuner supports an approximation tuning system
that automatically selects per-operation approximation knobs
to maximize program performance,
while allowing users to specify an acceptable degradation in accuracy.
ApproxHPVM and ApproxTuner provide an extensible system that researchers in compilers,
approximate computing and machine learning can use to optimize their applications,
and experiment with new custom approximation techniques.

Documentation
-------------

Please refer to :doc:`getting-started` for how to build and use HPVM.

.. toctree::
   :maxdepth: 1

   getting-started
   build-hpvm
   components/index
   specifications/index
   developerdocs/index
   gallery
   FAQs<faqs>

Indices and tables
------------------

* :ref:`genindex`

Support
-------

All questions can be directed to `hpvm-dev@lists.cs.illinois.edu <mailto:hpvm-dev@lists.cs.illinois.edu>`_.
