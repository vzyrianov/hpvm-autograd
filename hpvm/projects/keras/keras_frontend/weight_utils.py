
import sys
import numpy as np
import struct
import random
from keras.optimizers import Adam



def dumpLabels(file_name, Y_test, reload_weights = False):
    
    if reload_weights:
        return
    
    print ("Dumping Labels File = ", file_name)
    
    f = open(file_name, "wb")    
    labels_map = {}    
    for label in Y_test:
        label_val = 0
        if len(Y_test.shape) > 1:        
          label_val = np.int32(label[0])
        else:
          label_val = np.int32(label)
         
        if label_val not in labels_map:
            labels_map[label_val] = 0
            labels_map[label_val] += 1

        f.write(label_val)

    f.close()
    




def dumpData(file_name, X_test, reload_weights = False):

    if reload_weights:
        return
   
    print ("*Dumping Input File = ", file_name)
   
    f = open(file_name, "wb")

    X_test = X_test.flatten()
    X_test = X_test.astype(np.float32)
    X_test.tofile(f)    

    f.close()


  

def dumpConvWeights(file_name, X_test, N, C, H, W, reload_weights = False):

    if reload_weights:
        return
   
    print ("*Dumping Conv Weights to file = ", file_name)
   
    f = open(file_name, "wb")

    X_test = np.transpose(X_test, (3, 2, 0, 1))
    X_test = X_test.flatten()
    X_test = X_test.astype(np.float32)
    X_test.tofile(f)    

    f.close()


    
    
def dumpFcWeights(file_name, weights, H, W, reload_weights = False):

    if reload_weights:
        return
   
    print ("*Dumping FC weights to = ", file_name)
    
    f = open(file_name, "wb")
    for i in range(H):
        for j in range(W):
            f.write(weights[i][j])

    f.close()        


    
def dumpFcBias(file_name, bias, W, reload_weights = False):

    if reload_weights:
        return
   
    print ("*Dump Bias Weights = ", file_name)

    f = open(file_name, "wb")
    for i in range(W):
        f.write(bias[i])

    f.close()



def dumpCalibrationData(file_name, X_train, labels_fname, train_labels):

  combined_list = []
  for i in range(len(X_train)):
    tup = (X_train[i], train_labels[i])
    combined_list.append(tup)       
  
  np.random.shuffle(combined_list)
  #X_calibration = X_train[0:5000]

  data_list = []
  labels_list = []
  for i in range(5000):
    tup = combined_list[i]
    data_list.append(tup[0])
    labels_list.append(tup[1])

  data_list = np.array(data_list)
  labels_list = np.array(labels_list)
  
  dumpData(file_name, data_list)
  dumpLabels(labels_fname, labels_list)
  


def dumpCalibrationData2(file_name, test_data, labels_fname, test_labels):
   
  dumpData(file_name, test_data)
  dumpLabels(labels_fname, test_labels)
  
  


# Loads Existing HPVM FP32 weights
def reloadHPVMWeights(model, reload_dir, output_model):

  print ("***** Reloading pre-trained HPVM weights ****")
  
  for i in range(len(model.layers)):
    layer = model.layers[i]
    layer_name = layer.name
    #-- print ("*layer_name = ", layer_name)
    if "conv" in layer_name or "dense" in layer_name:
    
        w_path = reload_dir + layer_name + "_w.bin"
        #-- print ("** w_path = ", w_path)    
        w_arr = np.fromfile(w_path, dtype='float32')

        if layer.use_bias:
            b_path = reload_dir + layer_name + "_b.bin"
            b_arr = np.fromfile(b_path, dtype='float32')

        w_shape = layer.get_weights()[0].shape    
        if "conv" in layer_name:      
          w_nchw_shape = (w_shape[3], w_shape[2], w_shape[0], w_shape[1])      
          w_arr = np.reshape(w_arr, w_nchw_shape)
          w_arr = np.transpose(w_arr, (2,3,1,0))

        if "dense" in layer_name:      
          w_arr = np.reshape(w_arr, w_shape)

        if layer.use_bias:
            weights = [w_arr, b_arr]
        else:
            weights = [w_arr]
        
        layer.set_weights(weights)
        
    elif "batch_normalization" in layer_name:
        beta_path = reload_dir + layer_name + "_beta.bin"
        gamma_path = reload_dir + layer_name + "_gamma.bin"
        mean_path = reload_dir + layer_name + "_mean.bin"
        variance_path = reload_dir + layer_name + "_variance.bin"
        
        beta = np.fromfile(beta_path, dtype='float32')
        gamma = np.fromfile(gamma_path, dtype='float32')
        mean = np.fromfile(mean_path, dtype='float32')
        variance = np.fromfile(variance_path, dtype='float32')

        weights = [gamma, beta, mean, variance]
        
        layer.set_weights(weights)
            

  # Model recompilation needed after resetting weights
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.0001, decay=1e-6),
                metrics=['accuracy'])    

  return model
