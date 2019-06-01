import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow.eager as tfe
import pandas as pd
import numpy as np
# from sklearn.preprocessing
##

# tf.enable_eager_execution()


class UpSampleLayer(tf.keras.Model):
    def __init__(self, filer, **kwargs):
        super(UpSampleLayer, self).__init__()
        self.kernel = 4
        self.stride = 2
        self.padding = 'same'
        self.activation = 'relu'

        if 'kernel' in kwargs.keys():
            self.kernel = kwargs['kernel']
        if 'stride' in kwargs.keys():
            self.stride = kwargs['stride']
        if 'padding' in kwargs.keys():
            self.padding = kwargs['padding']
        if 'activation' in kwargs.keys():
            self.activation = kwargs['activation']

        self.conv = tf.keras.layers.Conv2DTranspose(filer, kernel=self.kernel, stride=self.stride,
                                                    padding=self.padding, activation=self.activation)

    def call(self, x):
        return self.conv(x)


class UpSampleBlock(tf.keras.Model):
    def __init__(self, filer, **kwargs):
        super(UpSampleBlock, self).__init__()
        self.cv1 = UpSampleLayer(128, kernel=4, stride=2, padding='same', activation='ReLU')
        self.cv2 = UpSampleLayer(64, kernel=4, stride=2, padding='same', activation='ReLU')
        self.cv3 = UpSampleLayer(3, kernel=7, stride=1, padding='same', activation='ReLU')

    def call(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x


ub = UpSampleBlock()
x = ub(img)


class Gen(tf.keras.Model):
    def __init__(self):
        super(Gen, self).__init__()

    def call(self, x):
        pass


class DisCommon(tf.keras.Model):
    def __init__(self):
        super(DisCommon, self).__init__()
        self.conv1 = ConvBlock(64)
        self.conv2 = ConvBlock(128)
        self.conv3 = ConvBlock(256)
        self.conv4 = ConvBlock(512)
        self.conv5 = ConvBlock(1024)
        self.conv6 = ConvBlock(2048)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class DisSrc(tf.keras.Model):
     def __init__(self):
         super(DisSrc, self).__init__()
         self.common = DisCommon()



     def call(self, x):
         pass
##

class DisCls(tf.keras.Model):
    def __init__(self):
        super(DisCls, self).__init__()
        pass

    def call(self, x):
        pass
##

d_src = DisSrc()
d_cls = DisCls()
g = Gen()

data_size = 100
img_width = 24
img_height = 24
img_channel = 3

hair_color = 3
gender = 1
old = 1

x_data = np.random.normal(size = [data_size, img_width, img_height, img_channel])
hair_color = np.random.uniform(low=0, high=3, size=[data_size])
gender = np.random.uniform(low=0, high = 1, size=[data_size])
old = np.random.uniform(low=0, high=1, size=[data_size])


##

train_dataset = tf.data.Dataset.from_tensor_slices(x_dat, y_data)

for img, label in train_dataset.take(2) :
    l_adv = np.log(d_src(img))

# define loss and optimizer


