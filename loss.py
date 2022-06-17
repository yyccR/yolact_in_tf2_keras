
import cv2
import numpy as np
import tensorflow as tf
from box_utils import iou, decode, match


class MultiBoxLoss:
    def __init__(self, batch_size, priors, bbox_alpha=1.5, pos_thresh=0.5, neg_thresh=0.4, masks_to_train=100):
        self.batch_size = batch_size
        # [num_priors, 4]
        self.priors = priors
        self.bbox_alpha = bbox_alpha
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.masks_to_train = masks_to_train

    def smooth_l1_loss(self, loc_p, loc_t):
        """计算目标边框损失"""
        diff = np.abs(loc_t - loc_p)
        # less_than_one = tf.cast(tf.keras.backend.less(diff, 1.0), dtype=tf.float32)
        less_than_one = np.array(diff < 1.0, dtype=np.float32)
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        loss = np.sum(loss)
        return loss

    def lincomb_mask_loss(self, gt_loc_all, gt_boxes, gt_masks, batch_positive_idx, batch_best_truth_idx, mask_proto, pred_masks):
        """ 计算mask损失
        :param loc_all: [batch, num_priors, 4(cx,cy,cw,ch)]
        :param gt_boxes: [batch, num_boxes, 4(x1,y1,x2,y2)]
        :param gt_masks: [batch, h, w, num_objs]
        :param batch_positive_idx: [batch, num_priors]
        :param batch_best_truth_idx: [batch, num_priors]
        :param mask_proto: [batch, 1/4, 1/4, 32]
        :param pred_masks: [batch, num_priors, 32]
        :return:
        """
        mask_h, mask_w = mask_proto.shape[1:3]

        for i in range(self.batch_size):
            # [h,w,num_objs] => [1/4, 1/4, num_objs] => [num_objs, 1/4, 1/4]
            gt_masks_resize = tf.image.resize(gt_masks[i], size=(mask_h, mask_w), method=tf.image.ResizeMethod.BILINEAR)
            gt_masks_resize = np.array(gt_masks_resize > 0.5, dtype=np.float32).transpose([2,0,1])

            # 正样本(conf>0)的索引
            pos_idx = batch_positive_idx[i]
            # 正样本边框(conf>0)的索引, 每个prior对应一个边框
            best_truth_pos_idx = batch_best_truth_idx[i][pos_idx]

            # pos_gt_box = pos_gt_decode(loc_all[i], self.priors)[pos_idx]
            pos_gt_box = gt_boxes[i][pos_idx]

            if pos_idx.size(0) == 0:
                continue

            cur_mask_proto = mask_proto[i]
            cur_mask_coef = pred_masks[i][pos_idx]

            if cur_mask_coef.shape[0] > self.masks_to_train:
                select = np.random.permutation(cur_mask_coef.shape[0])[:self.masks_to_train]
                cur_mask_coef = cur_mask_coef[select]
                pos_idx = pos_idx[select]
                pos_gt_box = pos_gt_box[select]

            # gt_masks_resize[]







    def __call__(self, inputs, gt_boxes, gt_labels, gt_masks, *args, **kwargs):
        """
        :param inputs: [box, cls, mask, mask_proto, semantic_seg_conv]
        :param gt_boxes: [batch, n, gt_boxes]
        :param gt_labels:
        :param gt_masks:
        :param args:
        :param kwargs:
        :return:
        """
        # pred_box: [batch,-1,4]
        # pred_cls: [batch, -1, cls_nums]
        # pred_mask: [batch,-1,mask_proto_channels]
        # pred_mask_proto: [batch, 1/4, 1/4, 32]
        # pred_semantic_seg: [batch, 1/8, 1/8, classes-1]
        pred_box, pred_cls, pred_mask, pred_mask_proto, pred_semantic_seg = inputs

        batch_loc = []
        batch_conf = []
        batch_prior_gt_boxes = []
        batch_best_truth_idx = []
        for i in range(self.batch_size):
            # 这里给每个prior找到对应的边框, overlap太小的不要, 类似于faster-rcnn/mask-rcnn里的anchors做法
            # loc:[num_prioris, 4(cx,cy,cw,ch)]
            # conf:[num_prioris]
            # best_truth_idx:[num_prioris]
            loc, conf, best_truth_idx = match(self.priors, gt_boxes[i], gt_labels[i], self.pos_thresh, self.neg_thresh)
            prior_gt_boxes = gt_boxes[i][best_truth_idx]

            batch_loc.append(loc)
            batch_conf.append(conf)
            batch_prior_gt_boxes.append(prior_gt_boxes)
            batch_best_truth_idx.append(best_truth_idx)

        loc_all = np.array(batch_loc, dtype=np.float32)
        conf_all = np.array(batch_conf, dtype=np.int16)
        prior_gt_boxes_all = np.array(batch_prior_gt_boxes, dtype=np.int32)
        batch_best_truth_idx = np.array(batch_best_truth_idx, dtype=np.int32)

        # 计算所有非背景边框损失
        pos = conf_all > 0
        box_loss = self.smooth_l1_loss(pred_box[pos], loc_all[pos])


