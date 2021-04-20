

import os
import subprocess

class Benchmark:

    def __init__(self, binary_path, test_accuracy):

        self.binary_path = binary_path
        self.test_accuracy = test_accuracy
        self.epsilon = 0.05 # Adding some slack for accuracy difference


    def getPath(self):
        return self.binary_path

    
    def readAccuracy(self, accuracy_file):

        f = open(accuracy_file, "r") # File with final benchmark accuracy 
        acc_str = f.read()
        return float(acc_str)
    
        
    def runKeras(self):

        # Test Bechmark accuracy with pretrained weights (hpvm_relaod)
        run_cmd = "python " + self.binary_path + " hpvm_reload "
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


    def runHPVM(self):

        # Test Bechmark accuracy with pretrained weights (hpvm_relaod)
        run_cmd = "python " + self.binary_path + " hpvm_reload frontend compile"
        try:
            subprocess.call(run_cmd, shell=True)
        except:
            return False

        working_dir = open("working_dir.txt").read()
        cur_dir = os.getcwd()
        
        os.chdir(working_dir)
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


    def runHPVMTests(self):

        for benchmark in self.benchmarks:
            test_success = benchmark.runHPVM()

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
            

        
            
if __name__ == "__main__":

    testMgr = BenchmarkTests()
    AlexNet = Benchmark("src/alexnet.py", 79.28)
    AlexNet_ImageNet = Benchmark("src/alexnet_imagenet.py", 56.30)
    AlexNet2 = Benchmark("src/alexnet2.py", 84.98)
    LeNet = Benchmark("src/lenet.py", 98.70)
    MobileNet = Benchmark("src/mobilenet_cifar10.py", 84.42)
    ResNet18 = Benchmark("src/resnet18_cifar10.py", 89.56)
    ResNet50 = Benchmark("src/resnet50_imagenet.py", 75.10)
    VGG16_cifar10 = Benchmark("src/vgg16_cifar10.py", 89.96)
    VGG16_cifar100 = Benchmark("src/vgg16_cifar100.py", 66.50)
    VGG16_ImageNet = Benchmark("src/vgg16_imagenet.py", 69.46)

    testMgr.addBenchmark(AlexNet)
    testMgr.addBenchmark(AlexNet_ImageNet)
    testMgr.addBenchmark(AlexNet2)
    testMgr.addBenchmark(LeNet)
    testMgr.addBenchmark(MobileNet)
    testMgr.addBenchmark(ResNet18)
    testMgr.addBenchmark(ResNet50)
    testMgr.addBenchmark(VGG16_cifar10)
    testMgr.addBenchmark(VGG16_cifar100)
    testMgr.addBenchmark(VGG16_ImageNet)

    testMgr.runKerasTests()
    testMgr.printKerasSummary()
    
    testMgr.runHPVMTests()
    testMgr.printHPVMSummary()

