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

from keras.datasets import cifar10
from Benchmark import Benchmark
from Config import MODEL_PARAMS_DIR


# Inherits from Benchmark class defined in src/Benchmark.py
class AlexNet_CIFAR10(Benchmark):

    # buildModel overrides the buildModel declared in src/Benchmark.py
    # Goal: Build a Keras Sequential Model (other model types not supported) and return the (uninitalized/untrained) Model 
    def buildModel(self):

        activation_type = 'tanh'
        weight_decay = 1e-4

        model = Sequential()
        
        model.add(Conv2D(64, (11, 11), padding='same', activation=activation_type,
                         kernel_regularizer=regularizers.l2(weight_decay), input_shape=(3, 32, 32)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(192, (5, 5), padding='same', activation=activation_type,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.3))

        model.add(Conv2D(384, (3, 3), padding='same', activation=activation_type, 
                       kernel_regularizer=regularizers.l2(weight_decay)))   
        model.add(Conv2D(256, (3, 3), padding='same', activation=activation_type, 
                       kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv2D(256, (3, 3), padding='same', activation=activation_type, 
                       kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        
        return model


    # This routine is called from the common `run` method in src/Benchmark.py
    # Goal: Return Training and Testing data after preprocessing/normalization
    def data_preprocess(self):

        (X_train, y_train), (X_val, y_val) = cifar10.load_data()

        X_train = X_train / 255.0
 
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / (std + 1e-7)

        X_test = np.fromfile(MODEL_PARAMS_DIR + '/alexnet_cifar10/test_input.bin', dtype=np.float32)
        y_test = np.fromfile(MODEL_PARAMS_DIR + '/alexnet_cifar10/test_labels.bin', dtype=np.uint32)

        X_test = X_test.reshape((-1,3,32,32))


        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/alexnet_cifar10/tune_input.bin', dtype=np.float32)
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/alexnet_cifar10/tune_labels.bin', dtype=np.uint32)

        X_tuner = X_tuner.reshape((-1,3,32,32))

        return X_train, y_train, X_test, y_test, X_tuner, y_tuner


    # Goal: Given a Keras Sequential Model - setup the training parameters, train, and return the trained Model
    def trainModel(self, model, X_train, y_train, X_test, y_test):

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001, decay=1e-6),
            metrics=['accuracy']
        )

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(X_train)


        def lr_schedule(epoch):
            lrate = 0.001
            if epoch > 20:
                lrate = 0.0005
            if epoch > 40:
                lrate = 0.0003
            if epoch > 60:
                lrate = 0.0001
            if epoch > 80:
                lrate = 0.00005  
            return lrate

        model.fit(
            X_train,
            y_train,
            batch_size=128,
            shuffle=True,
            epochs=100,
            validation_data=(X_test, y_test), 
            callbacks=[LearningRateScheduler(lr_schedule)]
        )

        return model


    
if __name__ == '__main__':

    # Using GPU ID 0 - Change to use different GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format - HPVM currently supports NCHW - NHWC format is not supported
    K.set_image_data_format('channels_first')


    # *** Below are Parameters specific to each benchmark *****
    reload_dir = MODEL_PARAMS_DIR + '/alexnet_cifar10/'
    ## Either the HPVM weights are loaded (above) or the Keras Model from the path below 
    keras_model_file = MODEL_PARAMS_DIR + '/keras/alexnet_cifar10.h5'
    data_dir = 'data/alexnet_cifar10_hpvm/'   # if reloading weights, data_dir can be set to empty string (value is ignored)
    src_dir = 'src/alexnet_cifar10_src_hpvm/'  # Directory where HPVM sources are downloaded
    num_classes = 10  # Specify num out output classes - CIFAR10 has `10` classes
    batch_size = 500  # Batch Size set to 500 - Adjust this value based on your GPU memory 

    # All Classes inherit from 'Benchmark` class in src/Benchmark.py and have a common Constructor
    AlexNet = AlexNet_CIFAR10('AlexNet_CIFAR10', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)

    # This invokes the common run function in src/Benchmark.py 
    AlexNet.exportToHPVM(sys.argv)

    
