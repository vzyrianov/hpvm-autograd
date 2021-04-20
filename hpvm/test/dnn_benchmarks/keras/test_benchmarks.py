

import os
import sys
import subprocess
import argparse
from Config import *


class Benchmark:

    def __init__(self, binary_path, output_dir, test_accuracy):

        self.binary_path = binary_path
        self.test_accuracy = test_accuracy
        self.output_dir = output_dir
        self.epsilon = 0.05 # Adding some slack for accuracy difference


    def getPath(self):
        return self.binary_path

    
    def readAccuracy(self, accuracy_file):

        f = open(accuracy_file, "r") # File with final benchmark accuracy 
        acc_str = f.read()
        return float(acc_str)
    
        
    def runKeras(self):

        # Test Bechmark accuracy with pretrained weights (hpvm_relaod)
        run_cmd = "python3 " + self.binary_path + " keras_reload "
        try:
            subprocess.call(run_cmd, shell=True)
        except:
            return False

        accuracy = self.readAccuracy("final_accuracy")

        print ("accuracy = ", accuracy, " test_accuracy = ", self.test_accuracy) 

        test_success = False
        if (abs(self.test_accuracy - accuracy) < self.epsilon):
            print ("Test for " + self. binary_path + " Passed ")
            test_success = True
        else:
            print ("Test Failed for " + self.binary_path)
            test_success = False

        return test_success


    def runHPVM(self, weights_dump):

        if weights_dump:
          # Test Benchmark with Keras weight dumping
          run_cmd = "python3 " + self.binary_path + " keras_reload frontend compile compile_tuner"
        else:
          # Test Benchmark accuracy with pretrained weights (hpvm_relaod)
          run_cmd = "python3 " + self.binary_path + " hpvm_reload frontend compile compile_tuner"
          
        try:
            subprocess.call(run_cmd, shell=True)
        except:
            return False

        #working_dir = open("working_dir.txt").read()
        cur_dir = os.getcwd()

        working_dir = self.output_dir 
        os.chdir(working_dir)

        print ("cur_dir = ", os.getcwd())
        
        binary_path =  "./HPVM_binary"
        
        try:
            subprocess.call(binary_path, shell=True)
        except:
            return False
        
        accuracy = self.readAccuracy("final_accuracy")
        print ("accuracy = ", accuracy, " test_accuracy = ", self.test_accuracy) 

        test_success = False
        if (abs(self.test_accuracy - accuracy) < self.epsilon):
            print ("Test for " + self. binary_path + " Passed ")
            test_success = True
        else:
            print ("Test Failed for " + self.binary_path)
            test_success = False

        os.chdir(cur_dir)  # Change back to original working directory
        
        return test_success


            
        

class BenchmarkTests:

    def __init__(self):

        self.benchmarks = []
        self.passed_tests = []
        self.failed_tests = []
        self.passed_hpvm_tests = []
        self.failed_hpvm_tests = []


    def addBenchmark(self, benchmark):

        self.benchmarks.append(benchmark)


    def runKerasTests(self):

        for benchmark in self.benchmarks:
            test_success = benchmark.runKeras()

            if not test_success:
                self.failed_tests.append(benchmark.getPath())
            else:
                self.passed_tests.append(benchmark.getPath())


    def runHPVMTests(self, weights_dump):

        for benchmark in self.benchmarks:
            test_success = benchmark.runHPVM(weights_dump)

            if not test_success:
                self.failed_hpvm_tests.append(benchmark.getPath())
            else:
                self.passed_hpvm_tests.append(benchmark.getPath())


    def printKerasSummary(self):

        failed_test_count = len(self.failed_tests)
        passed_test_count = len(self.passed_tests)
        
        print (" Tests Passed  = " + str(passed_test_count) + " / " + str(len(self.benchmarks)))
        print ("******* Passed Tests ** \n")
        for passed_test in self.passed_tests:
            print ("Passed: " + passed_test)

        print (" Tests Failed  = " + str(failed_test_count) + " / " + str(len(self.benchmarks)))
        print ("****** Failed Tests *** \n")
        for failed_test in self.failed_tests:
            print ("Failed: " + failed_test)
            

    # Returns False if any of the tests failed
    def printHPVMSummary(self):

        failed_test_count = len(self.failed_hpvm_tests)
        passed_test_count = len(self.passed_hpvm_tests)
        
        print (" Tests Passed  = " + str(passed_test_count) + " / " + str(len(self.benchmarks)))
        print ("******* Passed Tests ** \n")
        for passed_test in self.passed_hpvm_tests:
            print ("Passed: " + passed_test)

        print (" Tests Failed  = " + str(failed_test_count) + " / " + str(len(self.benchmarks)))
        print ("****** Failed Tests *** \n")
        for failed_test in self.failed_hpvm_tests:
            print ("Failed: " + failed_test)

        if failed_test_count > 0:
            return False

        return True


    
            
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--work-dir', type=str,
                        help='working dir for dumping frontend generated files')

    parser.add_argument('--dump-weights', action="store_true", help='dump h5 weights to bin (default: False)')
    args = parser.parse_args()

    work_dir = args.work_dir
    dump_weights = args.dump_weights

    #print (dump_weights)
    #sys.exit(0)
    
    if os.path.exists(work_dir):
        print ("Work Directory Exists. Delete it or use a different work directory.")
        sys.exit(0)
        
    os.mkdir(work_dir)
    os.chdir(work_dir)
    
    testMgr = BenchmarkTests()
    AlexNet = Benchmark(CUR_SRC_PATH + "/alexnet.py", "src/alexnet_cifar10_src_hpvm", 79.28)
    AlexNet_ImageNet = Benchmark(CUR_SRC_PATH + "/alexnet_imagenet.py", "src/alexnet_imagenet_src", 56.30)
    AlexNet2 = Benchmark(CUR_SRC_PATH + "/alexnet2.py", "src/alexnet2_cifar10_src", 84.98)
    LeNet = Benchmark(CUR_SRC_PATH + "/lenet.py", "src/lenet_mnist_src", 98.70)
    MobileNet = Benchmark(CUR_SRC_PATH + "/mobilenet_cifar10.py", "src/mobilenet_cifar10_src", 84.42)
    ResNet18 = Benchmark(CUR_SRC_PATH + "/resnet18_cifar10.py", "src/resnet18_cifar10_src", 89.56)
    ResNet50 = Benchmark(CUR_SRC_PATH + "/resnet50_imagenet.py", "src/resnet50_imagenet_src", 75.10)
    VGG16_cifar10 = Benchmark(CUR_SRC_PATH + "/vgg16_cifar10.py", "src/vgg16_cifar10_src", 89.96)
    VGG16_cifar100 = Benchmark(CUR_SRC_PATH + "/vgg16_cifar100.py", "src/vgg16_cifar100_src", 66.50)
    VGG16_ImageNet = Benchmark(CUR_SRC_PATH + "/vgg16_imagenet.py", "src/vgg16_imagenet_src", 69.46)

    #testMgr.addBenchmark(AlexNet)
    #testMgr.addBenchmark(AlexNet2)
    #testMgr.addBenchmark(LeNet)
    testMgr.addBenchmark(MobileNet)
    testMgr.addBenchmark(ResNet18)
    #testMgr.addBenchmark(ResNet50)
    #testMgr.addBenchmark(VGG16_cifar10)
    testMgr.addBenchmark(VGG16_cifar100)
    #testMgr.addBenchmark(VGG16_ImageNet)
    testMgr.addBenchmark(AlexNet_ImageNet)
  
    #testMgr.runKerasTests()
    #testMgr.printKerasSummary()
    
    testMgr.runHPVMTests(dump_weights)
    tests_passed = testMgr.printHPVMSummary()

    if not tests_passed:
        sys.exit(-1)
    
 
