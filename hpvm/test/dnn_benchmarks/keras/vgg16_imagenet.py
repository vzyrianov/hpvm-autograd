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

from Benchmark import Benchmark
from Config import MODEL_PARAMS_DIR



class VGG16(Benchmark):

    def buildModel(self):
        img_input = Input(shape=(3, 224, 224))

        # Block 1
        x = ZeroPadding2D(padding=(1, 1))(img_input)
        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 2
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(128, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(128, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 4
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Block 5
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #     x = Flatten(data_format='channels_first')(x)
        x = Flatten()(x)

        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096)(x)   
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1000)(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)
            
        return model


    def data_preprocess(self):
        X_train, y_train = None, None
        
        X_test = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_imagenet/test_input.bin', dtype=np.float32)
        X_test = X_test.reshape((-1, 3, 224, 224)) 
        y_test = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_imagenet/test_labels.bin', dtype=np.uint32)
        
        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_imagenet/tune_input.bin', dtype=np.float32)
        X_tuner = X_tuner.reshape((-1, 3, 224, 224)) 
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_imagenet/tune_labels.bin', dtype=np.uint32)
 
        return X_train, y_train, X_test, y_test, X_tuner, y_tuner
    
    
    def trainModel(self, model):

        assert False, "ImageNet training not supported - use Pretrained weights"


    
if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format
    K.set_image_data_format('channels_first')


    ### Parameters specific to each benchmark
    reload_dir = MODEL_PARAMS_DIR + '/vgg16_imagenet/'
    keras_model_file = MODEL_PARAMS_DIR + '/keras/vgg16_imagenet.h5'
    data_dir = 'data/vgg16_imagenet/' 
    src_dir = 'src/vgg16_imagenet_src/'
    num_classes = 1000
    batch_size = 25

    VGG16 = VGG16('VGG16_imagenet', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)
    
    VGG16.exportToHPVM(sys.argv)


    
