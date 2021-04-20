# HPVM Tensor Runtime


## Dependencies

- CUDA >= 9.1

- cuDNN >= 7

## Building Tensor Runtime

Tensor Runtime and the DNN sources using the Tensor runtime are built with the unified HPVM build system. These 
can also be separately built. HPVM Tensor Runtime can be built under the `build` directory as:

```
make -j ${NUM_THREADS} tensor_runtime
```

The tensor runtime is built as a static library under `build/lib/liibtensor_runtime.a` 

### TensorRT DNN Benchmarks

To assist development of tensor-based programs using only the tensor runtime, we include 
sources under `dnn_sources` that directly invoke the HPVM Tensor Runtime API calls for tensor operations, e.g., convolution, matrix multiplication, 
add, relu, among others.

Each benchmark can be build under your `build` directory as:

```
make -j ${NUM_THREADS} ${BENCHMARK}
``` 

Currently, 17 Benchmarks included:

|                        |                        |
|------------------------|------------------------|
| lenet_mnist_fp32       | lenet_mnist_fp16       |
| alexnet_cifar10_fp32   | alexnet_cifar10_fp16   |
| alexnet2_cifar10_fp32  | alexnet2_cifar10_fp16  |
| vgg16_cifar10_fp32     | vgg16_cifar10_fp16     |
| vgg16_cifar100_fp32    | vgg16_cifar100_fp16    |
| mobilenet_cifar10_fp32 | mobilenet_cifar10_fp16 |
| resnet18_cifar10_fp32  | resnet18_cifar10_fp16  |
| alexnet_imagenet_fp32  |                        |
| vgg16_imagenet_fp32    |                        |
| resnet50_imagenet_fp32 |                        |

`_fp32` suffix denotes fp32 binaries - these use the FP32 API calls 

`_fp_16` suffix denotes fp16 binaries - these use FP16 (half precision) calls.

