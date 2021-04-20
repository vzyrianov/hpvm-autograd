


import subprocess

def readAccuracy(f_path):

    f = open(f_path)
    acc_str = f.read()
    accuracy = float(acc_str)

    return accuracy


def printResults(results):

    for dnn in results:
        print ("DNN = ", dnn, " , Accuracy = ", results[dnn])


def createBaselineConfig(f_path, base_flag, num_layers):

    f = open(f_path, "w+")
    for i in range(num_layers):
        f.write(str(base_flag) + "\n")
    f.close()
    
        
if __name__ == "__main__":

    FP32_binary_paths = ["alexnet_cifar10_fp32", "alexnet2_cifar10_fp32", "resnet18_cifar10_fp32", "vgg16_cifar10_fp32", "vgg16_cifar100_fp32", "lenet_mnist_fp32", "mobilenet_cifar10_fp32"]
    FP16_binary_paths = ["alexnet_cifar10_fp16", "alexnet2_cifar10_fp16", "resnet18_cifar10_fp16", "vgg16_cifar10_fp16", "vgg16_cifar100_fp16", "lenet_mnist_fp16", "mobilenet_cifar10_fp16"]

    fp32_results = {}
    for binary_path in FP32_binary_paths:
        subprocess.call("./" + binary_path)
        accuracy = readAccuracy("final_accuracy")
        fp32_results[binary_path] = accuracy


    fp16_results = {}
    for binary_path in FP16_binary_paths:
        subprocess.call("./" + binary_path)
        accuracy = readAccuracy("final_accuracy")
        fp16_results[binary_path] = accuracy


    printResults(fp32_results)
    printResults(fp16_results)
