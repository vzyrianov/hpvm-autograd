import os
import sys
import glob

import numpy as np
import tensorflow as tf
import scipy
import scipy.io
import keras
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from keras.datasets import mnist
from Benchmark import Benchmark
from Config import MODEL_PARAMS_DIR



class LeNet_MNIST(Benchmark):

    def buildModel(self):

        # Network Compostion: 2 Conv Layers, 2 Dense Layers
        model = Sequential()

        # ConvLayer1
        model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='tanh', input_shape=(1, 28, 28)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # ConvLayer2
        model.add(Conv2D(64, (5, 5), activation='tanh', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        
        # DenseLayer1
        model.add(Dense(1024, activation='tanh'))
        # DenseLayer2
        
        model.add(Dense(self.num_classes, activation='tanh'))
        # Softmax Layer
        model.add(Activation('softmax'))

        return model


    def data_preprocess(self):
        (X_train, y_train), (X_val, y_val) = mnist.load_data()
        test_labels = y_val

        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_train = X_train.astype('float32')
        X_train /= 255

        X_test = np.fromfile(MODEL_PARAMS_DIR + '/lenet_mnist/test_input.bin', dtype=np.float32)
        X_test = X_test.reshape((-1, 1, 28, 28)) 
        y_test = np.fromfile(MODEL_PARAMS_DIR + '/lenet_mnist/test_labels.bin', dtype=np.uint32)
        
        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/lenet_mnist/tune_input.bin', dtype=np.float32)
        X_tuner = X_tuner.reshape((-1, 1, 28, 28)) 
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/lenet_mnist/tune_labels.bin', dtype=np.uint32)

        return X_train, y_train, X_test, y_test, X_tuner, y_tuner
    

    def trainModel(self, model, X_train, y_train, X_test, y_test):

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy']
        )

        model.fit(
            X_train, 
            y_train,
            batch_size=128,
            epochs=10,
            verbose=1,
            validation_data=(X_test, y_test)
        )
        
        return model
  

    
if __name__ == '__main__':
      
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format
    K.set_image_data_format('channels_first')


    ### Parameters specific to each benchmark
    reload_dir = MODEL_PARAMS_DIR + '/lenet_mnist/'
    keras_model_file = MODEL_PARAMS_DIR + '/keras/lenet_mnist.h5'
    data_dir = 'data/lenet_mnist/' 
    src_dir = 'src/lenet_mnist_src/'
    num_classes = 10
    batch_size = 500
    
    print (reload_dir)

    LeNet = LeNet_MNIST('LeNet_MNIST', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)
    
    LeNet.exportToHPVM(sys.argv)
