
from itertools import product
import math
import numpy as np
import tensorflow as tf
from yolact_backbone import ResnetBackbone
from yolact_head import HeadLayer

class Yolact:
    def __init__(self,
                 input_shape=[640, 640, 3],
                 num_classes=90,
                 is_training=True,
                 mask_proto_channels=32,
                 aspect_ratios=[1, 0.5, 2],
                 scales = [24, 48, 96, 192, 384],
                 conf_thres=0.05,
                 nms_thres=0.5,
                 max_num_detections=100):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_training = is_training
        self.mask_proto_channels = mask_proto_channels
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_num_detections =max_num_detections
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
                    w = scale * math.sqrt(ar) / self.input_shape[0]
                    # h = scale / ar / self.image_size
                    # This is for backward compatability with a bug where I made everything square by accident
                    h = w
                    priors.append([x, y, w, h])
        self.feature_prior_data.append(np.array(priors, dtype=np.float32))
        self.feature_prior_data = np.concatenate(self.feature_prior_data, axis=0)

    def nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * self.input_shape[0]

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = tf.range(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue

            preds = tf.concat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            # keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])

        idx = tf.concat(idx_lst, dim=0)
        classes = tf.concat(cls_lst, dim=0)
        scores = tf.concat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:self.max_num_detections]
        scores = scores[:self.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / self.input_shape[0], masks[idx], classes, scores


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
        head_cls_pred = tf.keras.layers.Softmax()(head_cls_pred)
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
        box = tf.concat([
            self.feature_prior_data[:,:2] + box_pred[:,:,:2] * 0.1 * self.feature_prior_data[:,2:],
            self.feature_prior_data[:,2:] * tf.exp(box_pred[:,:,2:] * 0.2)
        ], axis=-1)
        box_xy = box[:,2:] - box[:,2:] / 2
        box_wh = box[:,2:] + box_xy
        box = tf.concat([box_xy, box_wh], axis=-1)


    def build_graph(self):
        """
                       ↑------> mask_proto  ---↓
        构图 backbone->fpn----->   head      ----->
        :return:
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # [C3, C4, C5] shape:[1/8, 1/16, 1/32]
        backbones = self.backbone(inputs)
        # [P3, P4, P5, P6, P7] shape:[1/8, 1/16, 1/32, 1/64, 1/128]
        fpns = self.fpn(backbones)
        # mask shape:[batch, 1/4, 1/4, 32]
        mask_proto = self.proto(fpns[0])
        # [box, cls, mask] shape:[batch,-1,4],[batch, -1, cls_nums],[batch,-1,mask_proto_channels]
        heads = self.head(fpns)
        if self.is_training:
            # semantic segments shape:[batch, 1/8, 1/8, classes-1]
            semantic_seg_conv = tf.keras.layers.Conv2D(self.num_classes-1, kernel_size=1)(fpns[0])
            # box, cls, mask, semantic
            model = tf.keras.Model(inputs=inputs, outputs=[*heads, semantic_seg_conv])
        else:
            softmax_cls = tf.keras.layers.Softmax()(heads[1])


        return model


if __name__ == "__main__":
    yolact = Yolact()
    model = yolact.build_graph()
    model.summary(line_length=100)