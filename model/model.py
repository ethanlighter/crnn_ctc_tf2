from tensorflow import keras
from tensorflow.keras import layers
from model.resnet import resnet
import tensorflow as tf
from config import Config
class vgg(keras.Model):
    def __init__(self):
        super(vgg, self).__init__()
        self.conv1 = layers.Conv2D(64,3,padding="same",activation="relu")
        self.maxpool1 = layers.MaxPool2D(pool_size=2,padding="same")
        self.conv2 = layers.Conv2D(128,3,padding="same",activation="relu")
        self.maxpool2 = layers.MaxPool2D(pool_size=2,padding="same")
        self.conv3  = layers.Conv2D(256, 3, padding='same', use_bias=False)
        self.bacth3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.maxpool4 = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')

        self.conv5 = layers.Conv2D(512, 3, padding='same', use_bias=False)
        self.bacth5 = layers.BatchNormalization()
        self.relu5 = layers.Activation('relu')
        self.conv6 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.maxpool6 = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')

        self.conv7 = layers.Conv2D(512, 2, use_bias=False)
        self.batch7 = layers.BatchNormalization()
        self.relu7 = layers.Activation('relu')
    def call(self, inputs, training=False, mask=None):
        inputs = self.conv1(inputs)
        inputs = self.maxpool1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.maxpool2(inputs)
        inputs = self.conv3(inputs)
        inputs =self.bacth3(inputs,training=training)
        inputs =self.relu3(inputs)
        inputs = self.conv4(inputs)
        inputs = self.maxpool4(inputs)

        inputs =self.conv5(inputs)
        inputs =self.bacth5(inputs,training=training)
        inputs =self.relu5(inputs)
        inputs =self.conv6(inputs)
        inputs =self.maxpool6(inputs)

        inputs =self.conv7(inputs)
        inputs =self.batch7(inputs,training=training)
        inputs =self.relu7(inputs)
        return inputs


class vgg_crnn(keras.Model):
    def __init__(self):
        super(vgg_crnn, self).__init__()
        # self.backbone = vgg()
        self.backbone = resnet(layers_dim=[2,2,2,2])
        self.reshape_net = layers.Reshape((-1,512))
        self.lstm01 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))
        self.lstm02 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))
        self.fc1 = layers.Dense(units=Config.dict_size)

    def call(self, inputs, training=None, mask=None):
        inputs = self.backbone(inputs,training = training)
        inputs = self.reshape_net(inputs)
        inputs = self.lstm01(inputs)
        inputs = self.lstm02(inputs)
        return self.fc1(inputs)
#
#
# def vgg_style(input_tensor):
#     """
#     The original feature extraction structure from CRNN paper.
#     Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
#     """
#     x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
#     x = layers.MaxPool2D(pool_size=2, padding='same')(x)
#
#     x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
#     x = layers.MaxPool2D(pool_size=2, padding='same')(x)
#
#     x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
#     x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)
#
#     x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
#     x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)
#
#     x = layers.Conv2D(512, 2, use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     return x
#
#
# def build_model(num_classes, image_width=None, channels=1):
#     """build CNN-RNN model"""
#
#     img_input = keras.Input(shape=(32, image_width, channels))
#     x = vgg_style(img_input)
#     x = layers.Reshape((-1, 512))(x)
#
#     x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
#     x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
#     x = layers.Dense(units=num_classes)(x)
#     return keras.Model(inputs=img_input, outputs=x, name='CRNN')