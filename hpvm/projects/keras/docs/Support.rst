Supported Keras Operators
=========================

The Keras frontend supports ``Sequential()`` Keras models.
The list of supported operations is as follows:


* ``Conv2D``
* ``DepthwiseConv2D``
* ``Dense``
* ``BatchNormalization``
* ``MaxPooling2D``
* ``AveragePooling2D``
* ``Flatten``
* ``Add``
* ``ZeroPadding2D``
* ``Activation`` 

  * ``relu``
  * ``tanh``
  * ``softmax``

Limitations
-----------

* Currently, we support Convolutional Neural Networks (CNNs) that include the supported operators (above) - RNNs/LSTMs not supported
* We currently only support models in NCHW format (NHWC is not supported)
* Softmax operator should be the last operation in the CNN pipeline 
* Softmax operation must be a separate operator (not specified as activation to another type of Keras operator). Example of what works:

.. code-block:: python

   Activation ("softmax")

Example of what is NOT supported:

.. code-block:: python

   Dense(num_classes, activation="softmax")


* For convolutions with stride > 1 ``same`` convolution is NOT supported. Explicitly add ``ZeroPadding2D`` layer before ``Conv2D`` or ``DepthwiseConv2D`` operators. Example of what does NOT work:

.. code-block:: python

   Conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding="same")

Example of what works instead:

.. code-block:: python

   # NOTE: Amount of padding varies with kernel sizes and strides
   ZeroPadding2D(padding=(1, 1), data_format="channels_first") # only support NCHW
   Conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding="valid")
