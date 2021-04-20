
CNN Model Weights
===================

The CNN weights (and input) files can be downloaded from here: https://databank.illinois.edu/datasets/IDB-6565690

The extracted `model_params` directory is to be copied to `hpvm/hpvm/test/dnn_benchmarks/model_params` - the CNN benchmark expect the model weights at this specific location. The automatic HPVM install (`install.sh`) does the data download, extraction, and copying automatically.

We support CNN weights in 3 formats:

* `.h5` file format: The entire CNN model is stored as a single `.h5` file. The Keras models are shipped as `.h5` files. These can be found under `model_params/keras`

* `.pth.tar` file format: The PyTorch models are shipped as `pth.tar` files and can be found under `model_params/pytorch`

* `.bin` serialized binary file format: This format is used by the HPVM binaries. Convolution, Dense, and BatchNormalization parameters for each layer are stored as individual files. The weights are serialized FP32 values layed out serially in `NCHW` format. Our frontends (Keras and PyTorch) convert `.h5` and `pth.tar` files into `.bin` files in the frontend translation phase. The `.bin` weights can be found under the respective subdirectory for each benchmark under `model_params/`

