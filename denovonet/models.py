'''
models.py

Copyright (c) 2021 Karolis Sablauskas
Copyright (c) 2021 Gelana Khazeeva

This file is part of DeNovoCNN.

DeNovoCNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeNovoCNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
import os
import math

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, multiply, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2, l1

from keras.models import load_model

from denovonet.settings import NUMBER_CLASSES, BATCH_SIZE, MODEL_ARCHITECTURE, CLASS_MODE
from denovonet.settings import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

from denovonet.variants import TrioVariant, preprocess_image
from denovonet.dataset import CustomAugmentation

num_classes = NUMBER_CLASSES

def squeeze_excite_block2D(filters,inputs):
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, filters))(se) 
    se = Dense(filters//32, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = multiply([inputs, se])
    return se

def advanced_cnn_binary(input_shape, output_bias=None):
    # Based on https://github.com/Matuzas77/MNIST-0.17/blob/master/MNIST_final_solution.ipynb
    
    s = Input(shape=input_shape) 
    x = Conv2D(128,(3,3),activation='relu',padding='same')(s)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block2D(128,x)

    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block2D(128,x)
    x = AveragePooling2D(2)(x)


    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block2D(128,x)
    x = AveragePooling2D(2)(x)

    x = concatenate([GlobalMaxPooling2D()(x),
        GlobalAveragePooling2D()(x)])
    
    x = Dense(1, activation='sigmoid', kernel_regularizer=l1(0.00025), 
              kernel_initializer="he_normal")(x)
    
    return Model(inputs=s, outputs=x)

def cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def get_model(model_name, input_shape, num_classes):
    """
        Returns model
    """
    if model_name == 'cnn':
        model = cnn(input_shape, num_classes)
    elif model_name == 'advanced_cnn_binary':
        model = advanced_cnn_binary(input_shape)
    else:
        raise Exception('NameError','Unknown model name')

    return model

def get_number_images(images_directory):
    """
        Returns number of training images.
    """
    dnms_path = os.path.join(images_directory, 'dnm')
    ivs_path = os.path.join(images_directory, 'iv')
    print(dnms_path, len(os.listdir(dnms_path)))
    print(ivs_path, len(os.listdir(ivs_path)))

    return (len(os.listdir(dnms_path)) + len(os.listdir(ivs_path)))

def get_steps_per_epoch(images_directory):
    """
        Returns number of batches in one epoch.
    """
    total_images = get_number_images(images_directory)
    steps = math.ceil(total_images / BATCH_SIZE)

    return steps

def batch_preprocess(generator):
    """
    Generator of preprocessed batches.
    """
    for batch in generator:
        x, y = batch       
        yield preprocess_image(x), y
    
def train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, 
          output_model_path, continue_training=False, 
          input_model_path=None, aug_cfg={}):
    
    """
        Training DeNovoCNN.
    """
    
    train_folder = os.path.join(IMAGES_FOLDER, DATASET_NAME, 'train')
    val_folder = os.path.join(IMAGES_FOLDER, DATASET_NAME, 'val')

    train_steps = get_steps_per_epoch(train_folder)
    val_steps = get_steps_per_epoch(val_folder)
    
    print('train steps:',train_steps)
    print('val_steps:',val_steps)

    batch_size = BATCH_SIZE
    num_classes = NUMBER_CLASSES

    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
   
    # initialization of the model
    if continue_training:
        model = load_model(input_model_path)
    else:
        model = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
        
    model.summary()
    
    # optimizer
    adad = keras.optimizers.Adadelta()

    # learning schedule callback
    lrate = keras.callbacks.LearningRateScheduler(step_decay)

    callbacks_list = [lrate]

    # training data generator
    train_cfg = {"rescale": 1./255}
    for k,v in aug_cfg.items():
        train_cfg[k] = v
    
    train_datagen = ImageDataGenerator(**train_cfg)
    
    print ("Train ImageDataGenerator config:", train_cfg)
    
    # validation data generator
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # batch generators
    train_generator = batch_preprocess(train_datagen.flow_from_directory(
        train_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE))

    validation_generator = batch_preprocess(test_datagen.flow_from_directory(
            val_folder,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE))
    
    # loss 
    if CLASS_MODE == 'binary':
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.categorical_crossentropy
    
    # compile model
    model.compile(
        loss=loss,
        optimizer=adad,
        metrics=['accuracy', 'AUC'])
    
    # train model
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=callbacks_list,
    )
    
    # save model
    model.save(output_model_path)
    print('Model saved as : {}'.format(output_model_path))

    return model