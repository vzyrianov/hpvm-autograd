
Keras Frontend
==============

Install Keras Frontend after moving to directory ``/hpvm/hpvm/projects/keras``

Requirements
------------


* python == 3.6.x
* pip >= 18

If your system uses a different Python version, we recommend using the conda environment ``keras_python36.yml``. Install this using:

.. code-block::

   conda env create -f keras_python36.yml --name keras_python36

Activate the conda environment before installing the pip package (below) using:

.. code-block::

   conda activate keras_python36

**NOTE:** This step must be performed each time (for each shell process) the frontend is to be used.

Installing the Keras Frontend Package
-------------------------------------

At the root of this project (``/projects/keras/``) install the Keras frontend pip package as:

.. code-block::

   pip3 install -e ./

**NOTE:** If you are using the conda environment, activate it prior to this step.

Suppported Operations
---------------------

List of supported operations and limitations are documented
:doc:`here <keras-support>`.
