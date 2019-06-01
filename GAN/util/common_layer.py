import tensorflow as tf


class ConvTRLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvTRLayer, self).__init__()
        # set default conv attributes
        self.conv_dict = {}
        self.conv_dict['filter'] = 3
        self.conv_dict['kernel'] = 4
        self.conv_dict['strides'] = 2
        self.conv_dict['padding'] = 'same'
        self.conv_dict['activation'] = tf.nn.relu

        for key, value in kwargs.items():
            self.conv_dict[key] = value

        self.conv = tf.keras.layers.Conv2DTranspose(
            self.conv_dict['filter'],
            kernel_size=self.conv_dict['kernel'],
            strides=self.conv_dict['strides'],
            padding=self.conv_dict['padding'],
            activation=self.conv_dict['activation'])

    def call(self, x):
        # this for debug
        # print(self.conv_dict)
        return self.conv(x)


class ConvLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvLayer, self).__init__()
        # set default conv attributes
        self.conv_dict = {}
        self.conv_dict['filter'] = 3
        self.conv_dict['kernel'] = 4
        self.conv_dict['strides'] = 2
        self.conv_dict['padding'] = 'same'
        self.conv_dict['activation'] = tf.nn.relu

        for key, value in kwargs.items():
            self.conv_dict[key] = value

        self.conv = tf.keras.layers.Conv2D(
            self.conv_dict['filter'],
            kernel_size=self.conv_dict['kernel'],
            strides=self.conv_dict['strides'],
            padding=self.conv_dict['padding'],
            activation=self.conv_dict['activation'])

    def call(self, x):
        return self.conv(x)
