
import tensorflow as tf
from resnet_backbone import ResnetBackbone

class Yolact:
    def __init__(self, input_shape=[640, 640, 3]):
        self.input_shape = input_shape
        self.backbone = ResnetBackbone(input_shape=self.input_shape).build_graph()

    def backbone(self, input):
        """resnet backbone

        :param input: [batch, h, w, 3]
        :return: [C3, C4, C5] (shape:[1/8, 1/16, 1/32])
        """
        return self.backbone(input)

    def fpn(self, inputs):
        """ fpn layers
        :param inputs: [C3, C4, C5] (shape:[1/8, 1/16, 1/32])
        :return: outputs: [P3, P4, P5, P6, P7]
        """
        # 1/8, 1/16, 1/32
        C3, C4, C5 = inputs

        # lat_layers
        # 1/32
        C5_lat = tf.keras.layers.Conv2D(256, kernel_size=1)(C5)
        # 1/16
        C5_lat_upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')(C5_lat)
        C4_lat = C5_lat_upsample + tf.keras.layers.Conv2D(256, kernel_size=1)(C4)
        # 1/8
        C4_lat_upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')(C4_lat)
        C3_lat = C4_lat_upsample + tf.keras.layers.Conv2D(256, kernel_size=1)(C3)

        # pred layers
        # 1/32
        P5 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(C5_lat)
        P5 = tf.keras.layers.ReLU()(P5)
        # 1/16
        P4 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(C4_lat)
        P4 = tf.keras.layers.ReLU()(P4)
        # 1/8
        P3 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(C3_lat)
        P3 = tf.keras.layers.ReLU()(P3)

        # down sample
        # 1/64
        P6 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(P5)
        # 1/128
        P7 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(P6)

        return [P3, P4, P5, P6, P7]

    def proto(self, input):
        """ mask proto after fpn
        :param input: fpn_P3:1/8
        :return: P3_upsample:1/4
        """
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(input)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)

        # 这里原pytorch实现是interpolate插值
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)

        # 这里pytorch里将channel permute到了最后, tf里面本身就在最后,所以不用改
        x = tf.keras.layers.Conv2D(32, kernel_size=1)(x)
        # 这里原实现在make_net()方法里不用relu,但是在外面proto_net输出还是加了relu,所以加到这里
        x = tf.keras.layers.ReLU()(x)
        return x

    def head(self):
        """ head prediction after fpn.

        :return:
        """



    def build_graph(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # [C3, C4, C5] (shape:[1/8, 1/16, 1/32])
        backbones = self.backbone(inputs)
        # [P3, P4, P5, P6, P7] (shape:[1/8, 1/16, 1/32, 1/64, 1/128])
        fpns = self.fpn(backbones)
        # [batch, 1/4, 1/4, 32]
        mask_proto = self.proto(fpns[0])

        model = tf.keras.Model(inputs=inputs, outputs=fpns)
        return model


if __name__ == "__main__":
    yolact = Yolact()
    model = yolact.build_graph()
    model.summary(line_length=100)