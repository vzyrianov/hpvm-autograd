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

from keras.datasets import cifar100
from Benchmark import Benchmark
from Config import MODEL_PARAMS_DIR



class VGG16_CIFAR100(Benchmark):

    def buildModel(self):

        # Build the network of vgg for 100 classes 
        self.weight_decay = 0.0005
        self.x_shape = [3, 32, 32]

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def data_preprocess(self):

        (X_train, y_train), (X_val, y_val) = cifar100.load_data()

        X_train = X_train / 255.0
        #X_val = X_val / 255.0

        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / (std + 1e-7)
        #X_val = (X_val - mean) / (std + 1e-7)

        X_test = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_cifar100/test_input.bin', dtype=np.float32)
        y_test = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_cifar100/test_labels.bin', dtype=np.uint32)

        X_test = X_test.reshape((-1,3,32,32))

        X_tuner = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_cifar100/tune_input.bin', dtype=np.float32)
        y_tuner = np.fromfile(MODEL_PARAMS_DIR + '/vgg16_cifar100/tune_labels.bin', dtype=np.uint32)

        X_tuner = X_tuner.reshape((-1,3,32,32))

        return X_train, y_train, X_test, y_test, X_tuner, y_tuner


    def trainModel(self,model, X_train, y_train, X_test, y_test):

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        batch_size = 128
        learning_rate = 0.1
        lr_drop = 30


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)


        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=learning_rate),
            metrics=['accuracy']
        )
        
        # training process in a for loop with learning rate drop every 25 epoches.
        
        model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=X_train.shape[0] // batch_size,
            epochs=250,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr]
        )

        return model


    
if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Changing to NCHW format
    K.set_image_data_format('channels_first')


    ### Parameters specific to each benchmark
    reload_dir = MODEL_PARAMS_DIR + '/vgg16_cifar100/'
    keras_model_file = MODEL_PARAMS_DIR + '/keras/vgg16_cifar100.h5'
    data_dir = 'data/vgg16_cifar100/' 
    src_dir = 'src/vgg16_cifar100_src/'
    num_classes = 100
    batch_size = 100

    VGG16 = VGG16_CIFAR100('VGG16_CIFAR100', reload_dir, keras_model_file, data_dir, src_dir, num_classes, batch_size)
    
    VGG16.exportToHPVM(sys.argv)
    
