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



class MobileNet_CIFAR10(Benchmark):

    def buildModel(self):
        alpha=1
        depth_multiplier=1

        model = Sequential()

        def _conv_block(filters, alpha, kernel=(3, 3), strides=(1, 1)):
            channel_axis = 1

            model.add(Conv2D(filters, kernel,
                              padding='same',
                              use_bias=False,
                              strides=strides, 
                              input_shape=(3, 32, 32)))
            model.add(BatchNormalization(axis=channel_axis))
            model.add(Activation('relu'))

        def _depthwise_conv_block(pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1)):
            channel_axis = 1 

            model.add(ZeroPadding2D(padding=((1,1), (1,1))))

            model.add(DepthwiseConv2D((3, 3),
                                       padding='valid',
                                       #depth_multiplier=depth_multiplier,
                                       strides=strides,
                                       use_bias=False))    
            model.add(BatchNormalization(axis=channel_axis))

            model.add(Activation('relu'))
            model.add(Conv2D(pointwise_conv_filters, (1, 1),
                              padding='same',
                              use_bias=False,
                              strides=(1, 1)))
            model.add(BatchNormalization(axis=channel_axis))
            model.add(Activation('relu'))


        _conv_block(32, alpha, strides=(1, 1))

        _depthwise_conv_block(64, alpha, depth_multiplier)

        _depthwise_conv_block(128, alpha, depth_multiplier,
                                  strides=(2, 2))
        _depthwise_conv_block(128, alpha, depth_multiplier)
        model.add(Dropout(rate=0.5))

        _depthwise_conv_block(256, alpha, depth_multiplier, 
                          strides=(2, 2))
        _depthwise_conv_block(256, alpha, depth_multiplier)
        model.add(Dropout(rate=0.5))

        _depthwise_conv_block(512, alpha, depth_multiplier,
                          strides=(2, 2))
        _depthwise_conv_block(512, alpha, depth_multiplier)
        _depthwise_conv_block(512, alpha, depth_multiplier)
        model.add(Dropout(rate=0.5))

        _depthwise_conv_block(512, alpha, depth_multiplier)
        _depthwise_conv_block(512, alpha, depth_multiplier)
        _depthwise_conv_block(512, alpha, depth_multiplier)
        model.add(Dropout(rate=0.5))

        _depthwise_conv_block(1024, alpha, depth_multiplier,
                             strides=(2, 2))
        _depthwise_conv_block(1024, alpha, depth_multiplier)
        model.add(Dropout(rate=0.5))

        model.add(AveragePooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(self.num_classes))    
        model.add(Activation('softmax'))

        return model

    
    def data_preprocess(self):

        (X_train, y_train), (X_val, y_val) = cifar10.load_data()

        X_train = X_train / 255.0
        #X_val = X_val / 255.0

        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / (std + 1e-7)
        #X_val = (X_val - mean) / (std + 1e-7)

        X_test = np.fromfile(MODEL_PARAMS_DIR + '/mobilenet_cifar10/test_input.bin', dtype=np.float32)
        y_test= np.fromfile(MODEL_PARAMS_DIR + '/mobilenet_cifar10/test_labels.bin', dtype=np.uint32)

        X_test = X_test.reshape((-1,3,32,32))

        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/mobilenet_cifar10/tune_input.bin', dtype=np.float32)
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/mobilenet_cifar10/tune_labels.bin', dtype=np.uint32)

        X_tuner = X_tuner.reshape((-1,3,32,32))


        return X_train, y_train, X_test, y_test, X_tuner, y_tuner


    def trainModel(self, model, X_train, y_train, X_test, y_test):

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        # data augmentation, horizontal flips only
        datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=0.0,
                width_shift_range=0.0,
                height_shift_range=0.0,
                vertical_flip=False,
                horizontal_flip=True)
        datagen.fit(X_train)


        learning_rates=[]
        for i in range(50):
            learning_rates.append(0.01)
        for i in range(75-50):
            learning_rates.append(0.001)
        for i in range(100-75):
            learning_rates.append(0.0001)
        for i in range(125-100):
            learning_rates.append(0.00001)
            
        callbacks = [
            LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
        ]

        model.compile(optimizer=keras.optimizers.SGD(lr=learning_rates[0], momentum=0.9, decay=0.0), 
                               loss='categorical_crossentropy', 
                               metrics=['accuracy'])

        model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=128),
            steps_per_epoch=int(np.ceil(50000 / 128)),
            validation_data=(X_test, y_test),
            epochs=125,
            callbacks=callbacks
        )

        return model

  
    
if __name__ == '__main__':
      
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format
    K.set_image_data_format('channels_first')


    ### Parameters specific to each benchmark
    reload_dir = MODEL_PARAMS_DIR + '/mobilenet_cifar10/'
    keras_model_file = MODEL_PARAMS_DIR + '/keras/mobilenet_cifar10.h5'
    data_dir = 'data/mobilenet_cifar10/' 
    src_dir = 'src/mobilenet_cifar10_src/'
    num_classes = 10
    batch_size = 500

    MobileNet = MobileNet_CIFAR10('MobileNet_CIFAR10', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)
    
    MobileNet.exportToHPVM(sys.argv)

