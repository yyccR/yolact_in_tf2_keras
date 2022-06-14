
import tensorflow as tf


class HeadLayer(tf.keras.layers.Layer):
    def __init__(self,aspect_ratios, num_classes, mask_proto_channels):
        super(HeadLayer, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.num_classes = num_classes
        self.mask_proto_channels = mask_proto_channels
        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')
        self.relu1 = tf.keras.layers.ReLU()
        self.conv_box = tf.keras.layers.Conv2D(len(self.aspect_ratios)*4,kernel_size=3, padding='same')
        self.conv_cls = tf.keras.layers.Conv2D(len(self.aspect_ratios)*self.num_classes,kernel_size=3, padding='same')
        self.conv_mask = tf.keras.layers.Conv2D(len(self.aspect_ratios)*self.mask_proto_channels,kernel_size=3, padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        box = self.conv_box(x)
        box = tf.keras.layers.Reshape([-1, 4])(box)
        cls = self.conv_cls(x)
        cls = tf.keras.layers.Reshape([-1, self.num_classes])(cls)
        mask = self.conv_mask(x)
        mask = tf.keras.activations.tanh(mask)
        mask = tf.keras.layers.Reshape([-1, self.mask_proto_channels])(mask)
        return [box, cls, mask]