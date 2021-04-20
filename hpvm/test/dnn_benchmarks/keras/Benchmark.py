

import sys
import os
import shutil
import subprocess
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras_frontend.approxhpvm_translator import translate_to_approxhpvm
from keras_frontend.weight_utils import dumpCalibrationData
from keras_frontend.weight_utils import reloadHPVMWeights


# Every CNN Benchmark must inherit from Benchmark class
# Defines common interfaces and virtual methods to be overridden by child classes
class Benchmark:

    def __init__(self, name, reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size=500):
        self.name = name
        self.reload_dir = reload_dir + "/"
        self.keras_model_file = keras_model_file
        self.data_dir = data_dir + "/"
        self.src_dir = src_dir + "/"
        self.num_classes = num_classes
        self.batch_size = batch_size
        
    # Override function in subclass    
    def buildModel(self):
        return

    # Override function in subclass
    def data_preprocess(self):
        return

    # Override function in subclass
    def trainModel(self, X_train, y_train, X_test, y_test):
        return

    # Override function in subclass
    def inference(self):
        return


    # Common Function - Do not override in Subclasses
    def compileSource(self, working_dir, src_name, binary_name):
              
        src_file = os.getcwd() + "/" + working_dir + "/" + src_name   #  approxhpvm_src.cc"
        target_binary = os.getcwd() + "/" + working_dir + "/" + binary_name    # HPVM_binary"
        approx_conf_file = os.getcwd() + "/" + working_dir + "/tuner_confs.txt"

        FNULL = open(os.devnull, 'w')
        
        try:
            subprocess.run([
                "hpvm-clang", 
                "-h"
            ], check=True, stdout=FNULL)
            
        except:
            print("""

            ERROR: Could not find hpvm-clang (HPVM compile script)!!

            hpvm-clang is installed to the python environment used when compiling HPVM.
            Please try rerunning 'make -j hpvm-clang' and make sure `hpvm-clang` is in your $PATH""")
            
            sys.exit(1)

        try:
            subprocess.run([
                "hpvm-clang", src_file, target_binary,
                "-t", "tensor", "--conf-file", approx_conf_file
            ], check=True)
        except:
            print ("\n\n ERROR: HPVM Compilation Failed!! \n\n")
            sys.exit(1)

        f = open("working_dir.txt", "w+")
        f.write(working_dir)
        f.close()
       
            
        
    def printUsage(self):

        print ("Usage: python ${benchmark.py} [hpvm_reload|train] [frontend] [compile]")
        sys.exit(0)

    # Common Function for Exporting to HPVM Modules - Do not Override in Subclasses    
    def exportToHPVM(self, argv):

      if len(argv) < 2:
          self.printUsage()
          
      print ("Build Model ...")
      # Virtual method call implemented by each CNN
      model = self.buildModel()

      print ("Data Preprocess... \n")
      # Virtual method call to preprocess test and train data 
      X_train, y_train, X_test, y_test, X_tuner, y_tuner = self.data_preprocess()   

      if argv[1] == "hpvm_reload":
        print ("loading weights .....\n\n")  
        model = reloadHPVMWeights(model, self.reload_dir, self.keras_model_file)

      elif argv[1] == "keras_reload":
        model.load_weights(self.keras_model_file)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])   

      elif argv[1] == "train":
        print ("Train Model ...")
        model = self.trainModel(model, X_train, y_train, X_test, y_test)
      else:
          self.printUsage()

          
      score = model.evaluate(X_test, to_categorical(y_test, self.num_classes), verbose=0)
      print('Test accuracy2:', score[1])

      f = open("final_accuracy", "w+")
      f.write(str(score[1] * 100))
      f.close()


      if len(argv) > 2:
        if argv[2] == "frontend":

          if not os.path.isabs(self.data_dir):
              self.data_dir = os.getcwd() + "/" + self.data_dir
           
          if argv[1] == "hpvm_reload": # If reloading HPVM weights use this as directory to load from in HPVM-C generated src
              self.data_dir = self.reload_dir
          
          # Main call to ApproxHPVM-Keras Frontend
          working_dir = translate_to_approxhpvm(model,
                                                self.data_dir, self.src_dir,   
                                                X_test, y_test,
                                                X_tuner, y_tuner,
                                                self.batch_size, # FIXIT
                                                self.num_classes,
                                                (argv[1] == "hpvm_reload")) # Do not redump HPVM weights if `hpvm_reload` used

          if len(argv) > 3:
            if argv[3] == "compile":
              self.compileSource(working_dir, "approxhpvm_src.cc", "HPVM_binary")
            else:
              self.printUsage()

          if len(argv) > 4:
            if argv[4] == "compile_tuner":
              self.compileSource(working_dir, "approxhpvm_tuner_src.cc", "HPVM_tuner_binary")
            else:
              self.printUsage()


        if argv[2] == "keras_dump":
          model.save_weights(self.keras_model_file)

          

    

        
