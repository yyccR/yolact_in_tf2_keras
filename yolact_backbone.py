

import tensorflow as tf

class BottleneckLayer(tf.keras.layers.Layer):
    def __init__(self,
                 output_channels,
                 strides=1,
                 downsample=None,
                 norm_layer=tf.keras.layers.BatchNormalization,
                 dilation=1):
        super(BottleneckLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=1, use_bias=False,
                                            dilation_rate=dilation)
        self.bn1 = norm_layer()
        self.conv2 = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=3, strides=strides, padding='same',
                                            use_bias=False, dilation_rate=dilation)
        self.bn2 = norm_layer()
        self.conv3 = tf.keras.layers.Conv2D(filters=output_channels * 4, kernel_size=1, use_bias=False,
                                            dilation_rate=dilation)
        self.bn3 = norm_layer()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = strides

    def call(self, inputs, *args, **kwargs):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        final_out = out + residual
        final_out = self.relu(final_out)

        return final_out


class ResnetBackbone:
    def __init__(self,
                 input_shape=[640, 640, 3],
                 layers=[3, 4, 23, 3],
                 atrous_layers=[],
                 block=BottleneckLayer,
                 norm_layer=tf.keras.layers.BatchNormalization):
        self.input_shape = input_shape
        self.num_base_layers = len(layers)
        self.layers = layers
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers
        self.inplanes = 64
        self.block = block

    def _make_layer(self, output_channels, blocks, stride=1):
        """bottleneck layers"""
        downsample = None
        if stride != 1 or self.inplanes != output_channels * 4:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=output_channels * 4,
                                       kernel_size=1,
                                       strides=stride,
                                       use_bias=False,
                                       dilation_rate=self.dilation),
                self.norm_layer()
            ])

        layers = tf.keras.Sequential()
        layers.add(self.block(output_channels, stride, downsample, self.norm_layer, self.dilation))
        self.inplanes = output_channels * 4
        for i in range(1, blocks):
            layers.add(self.block(output_channels, norm_layer=self.norm_layer))
        self.channels.append(output_channels * 4)
        return layers

    def build_graph(self):
        """resnet"""
        input = tf.keras.layers.Input(shape=self.input_shape)
        # 1/2
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # 1/4
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        # 1/4
        C2 = x = self._make_layer(output_channels=64, blocks=self.layers[0])(x)
        # 1/8
        C3 = x = self._make_layer(output_channels=128, blocks=self.layers[1], stride=2)(x)
        # 1/16
        C4 = x = self._make_layer(output_channels=256, blocks=self.layers[2], stride=2)(x)
        # 1/32
        C5 = x = self._make_layer(output_channels=512, blocks=self.layers[3], stride=2)(x)
        model = tf.keras.models.Model(inputs=input, outputs=[C3, C4, C5])
        return model

if __name__ == "__main__":
    resnet = ResnetBackbone()
    model = resnet.build_graph()
    model.summary(line_length=100)