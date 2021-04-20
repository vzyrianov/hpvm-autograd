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



class ResNet50(Benchmark):
    
    def buildModel(self):
        
        def identity_block(input_tensor, kernel_size, filters, stage, block):
            filters1, filters2, filters3 = filters
            bn_axis = 1

            x = Conv2D(filters1, (1, 1))(input_tensor)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters2, kernel_size,
                              padding='same')(x)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters3, (1, 1))(x)
            x = BatchNormalization(axis=bn_axis)(x)

            x = add([x, input_tensor])
            x = Activation('relu')(x)
            return x

        def conv_block(input_tensor,
                       kernel_size,
                       filters,
                       stage,
                       block,
                       strides=(2, 2)):
            filters1, filters2, filters3 = filters
            bn_axis = 1
            x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters2, kernel_size, padding='same')(x)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters3, (1, 1))(x)
            x = BatchNormalization(axis=bn_axis)(x)

            shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
            shortcut = BatchNormalization(
                axis=bn_axis)(shortcut)

            x = add([x, shortcut])
            x = Activation('relu')(x)
            return x

        img_input = Input(shape=(3, 224, 224))
        bn_axis = 1

        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    #     x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = BatchNormalization(axis=bn_axis)(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7))(x)
        x = Flatten()(x)
        x = Dense(1000)(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)
        
        return model

    
    def data_preprocess(self):
        X_train, y_train = None, None
        
        X_test = np.fromfile(MODEL_PARAMS_DIR + '/resnet50_imagenet/test_input.bin', dtype=np.float32)
        X_test = X_test.reshape((-1, 3, 224, 224)) 
        y_test = np.fromfile(MODEL_PARAMS_DIR + '/resnet50_imagenet/test_labels.bin', dtype=np.uint32)
        
        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/resnet50_imagenet/tune_input.bin', dtype=np.float32)
        X_tuner = X_tuner.reshape((-1, 3, 224, 224)) 
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/resnet50_imagenet/tune_labels.bin', dtype=np.uint32)
 
        return X_train, y_train, X_test, y_test, X_tuner, y_tuner
    

    def trainModel(self, model):

        assert False, "ImageNet training not supported - use Pretrained weights"


    
if __name__ == '__main__':
      
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format
    K.set_image_data_format('channels_first')


    ### Parameters specific to each benchmark
    reload_dir = MODEL_PARAMS_DIR + '/resnet50_imagenet/'
    keras_model_file = MODEL_PARAMS_DIR + '/keras/resnet50_imagenet.h5'
    data_dir = 'data/resnet50_imagenet/' 
    src_dir = 'src/resnet50_imagenet_src/'
    num_classes = 1000
    batch_size = 50

    ResNet50 = ResNet50('ResNet50_imagenet', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)
    
    ResNet50.exportToHPVM(sys.argv)


    
