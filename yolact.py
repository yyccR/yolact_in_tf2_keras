
from itertools import product
import numpy as np
import tensorflow as tf
from yolact_backbone import ResnetBackbone
from yolact_head import HeadLayer

class Yolact:
    def __init__(self,
                 input_shape=[640, 640, 3],
                 num_classes=90,
                 mask_proto_channels=32,
                 aspect_ratios=[1, 0.5, 2],
                 scales = [24, 48, 96, 192, 384]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mask_proto_channels = mask_proto_channels
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.feature_prior_data = []
        # 预生成基础坐标信息
        self.make_priors()
        self.backbone = ResnetBackbone(input_shape=self.input_shape).build_graph()
        self.head = HeadLayer()

    def make_priors(self):
        """预设坐标xywh, 类似于yolo"""
        features_size = [
            [self.input_shape[0] / 8, self.input_shape[1] / 8],
            [self.input_shape[0] / 16, self.input_shape[1] / 16],
            [self.input_shape[0] / 32, self.input_shape[1] / 32],
            [self.input_shape[0] / 64, self.input_shape[1] / 64],
            [self.input_shape[0] / 128, self.input_shape[1] / 128],
        ]
        for i, (feature_h, feature_w) in enumerate(features_size):
            priors = []
            scale = self.scales[i]
            for j, i in product(range(feature_h), range(feature_w)):
                x = (i + 0.5) / feature_w
                y = (j + 0.5) / feature_h
                for ar in self.aspect_ratios:
                    w = scale * ar / self.input_shape[0]
                    # h = scale / ar / self.image_size
                    # This is for backward compatability with a bug where I made everything square by accident
                    h = w
                    priors.append([x, y, w, h])
        self.feature_prior_data.append(np.array(priors, dtype=np.float32))
        self.feature_prior_data = np.concatenate(self.feature_prior_data, axis=0)

    def backbone(self, input):
        """resnet backbone

        :param input: [batch, h, w, 3]
        :return: [C3, C4, C5] (shape:[1/8, 1/16, 1/32])
        """
        return self.backbone(input)

    def fpn(self, inputs):
        """ fpn layers
        :param inputs: [C3, C4, C5] (shape:[1/8, 1/16, 1/32])
        :return: outputs: [P3, P4, P5, P6, P7] (shape:[1/8, 1/16, 1/32, 1/64, 1/128])
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
        x = tf.keras.layers.Conv2D(self.mask_proto_channels, kernel_size=1)(x)
        # 这里原实现在make_net()方法里不用relu,但是在外面proto_net输出还是加了relu,所以加到这里
        x = tf.keras.layers.ReLU()(x)
        return x

    def head(self, inputs):
        """ head prediction after fpn.
        :param inputs: [P3, P4, P5, P6, P7]
        :return:[box, cls, mask] (shape:[batch,-1,4],[batch, -1, cls_nums],[batch,-1,mask_proto_channels])
        """
        P3, P4, P5, P6, P7 = inputs
        head3 = self.head(P3)
        head4 = self.head(P4)
        head5 = self.head(P5)
        head6 = self.head(P6)
        head7 = self.head(P7)

        head_box_pred = tf.keras.layers.Concatenate(axis=-2)([
            head3[0], head4[0], head5[0], head6[0], head7[0]
        ])
        head_cls_pred = tf.keras.layers.Concatenate(axis=-2)([
            head3[1], head4[1], head5[1], head6[1], head7[1]
        ])
        head_mask_pred = tf.keras.layers.Concatenate(axis=-2)([
            head3[2], head4[2], head5[2], head6[2], head7[2]
        ])
        return [head_box_pred, head_cls_pred, head_mask_pred]

    def detect(self, box_pred, cls_pred, mask_pred, proto_pred):
        """
        :param box_pred: [batch, num_priors, 4]
        :param cls_pred: [batch, num_priors, num_classes]
        :param mask_pred: [batch, num_priors, mask_dim]
        :param proto_pred: [batch, mask_h, mask_w, mask_dim]
        :return:  class idx, confidence, bbox coords, mask
                 (batch_size, top_k, 1 + 1 + 4 + mask_dim)
        """



    def build_graph(self):
        """
                       ↑------> mask_proto  ---↓
        构图 backbone->fpn----->   head      ------>
        :return:
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # [C3, C4, C5] shape:[1/8, 1/16, 1/32]
        backbones = self.backbone(inputs)
        # [P3, P4, P5, P6, P7] shape:[1/8, 1/16, 1/32, 1/64, 1/128]
        fpns = self.fpn(backbones)
        # mask shape:[batch, 1/4, 1/4, 32]
        mask_proto = self.proto(fpns[0])
        semantic_seg_conv = tf.keras.layers.Conv2D(self.num_classes-1, kernel_size=1)(fpns[0])
        # [box, cls, mask] shape:[batch,-1,4],[batch, -1, cls_nums],[batch,-1,mask_proto_channels]
        heads = self.head(fpns)

        model = tf.keras.Model(inputs=inputs, outputs=fpns)
        return model


if __name__ == "__main__":
    yolact = Yolact()
    model = yolact.build_graph()
    model.summary(line_length=100)