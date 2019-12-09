import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf

class TransformLayer(Layer):

    def __init__(self, **kwargs):
        super(TransformLayer, self).__init__(**kwargs)
        self.activation = activations.get("relu")
    
    def build(self, input_shape):

        xAxis = input_shape[1]
        yAxis = input_shape[2]
        zAxis = input_shape[3]

        total = xAxis*yAxis*zAxis

        self.kernel = self.add_weight(name='kernel', 
                                  shape=(total,1),
                                  initializer='he_normal',
                                  trainable=True)

        self.bias = self.add_weight(name='bias', 
                                   shape=(total),
                                   initializer='zeros',
                                   trainable=True)

        super(TransformLayer,self).build(input_shape)

    def call(self, x):
        xAxis = x.shape[1]
        yAxis = x.shape[2]
        zAxis = x.shape[3]

        total = xAxis*yAxis*zAxis
        result = keras.reshape(x, [-1,total])
        
        result = keras.dot(result,self.kernel)+self.bias
        result = self.activation(result)

        result = keras.reshape(result,[-1,xAxis,yAxis,zAxis])
        
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


def unet(pretrained_weights = None,input_size = (256,256,1),model_type = 0):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)  

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)   

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    
    
    if model_type == 1:
        x21 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        x21 = concatenate([conv3,x21],axis = 3)
        x21 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x21)
        x21 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x21)

        x12 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x21))
        x12 = concatenate([conv2,x12],axis = 3)
        x12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x12)
        x12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x12)

        x03 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x12))
        x03 = concatenate([conv1,x03],axis = 3)
        x03 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x03)
        x03 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x03)   

    if model_type == 2:
        x21 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        x21 = concatenate([conv3,x21],axis = 3)
        x21 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x21)
        x21 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x21) 

        x11 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
        x11 = concatenate([conv2,x11],axis = 3)
        x11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x11)
        x11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x11)

        x12 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x21))
        x12 = concatenate([x11,x12],axis = 3)
        x12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x12)
        x12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x12)

        x01 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
        x01 = concatenate([conv1,x01],axis = 3)
        x01 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x01)
        x01 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x01)

        x02 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x11))
        x02 = concatenate([x02,x01],axis = 3)
        x02 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x02)
        x02 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x02)

        x03 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x12))
        x03 = concatenate([x02,x03],axis = 3)
        x03 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x03)
        x03 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x03)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    
    merge7 = concatenate([conv3,up7], axis = 3)

    if model_type!=0:
        merge7 = concatenate([up7,x21],axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    if model_type!=0:
        merge8 = concatenate([up8,x12],axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)
    
    if model_type!=0:
        merge9 = concatenate([up9,x03],axis = 3)
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model