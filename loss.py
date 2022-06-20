
import cv2
import numpy as np
import tensorflow as tf
from box_utils import iou, decode, match


class MultiBoxLoss:
    def __init__(self, batch_size, priors, conf_alpha=1, bbox_alpha=1.5, mask_alpha=6.125, pos_thresh=0.5, neg_thresh=0.4, masks_to_train=100):
        self.batch_size = batch_size
        # [num_priors, 4]
        self.priors = priors
        self.conf_alpha = conf_alpha
        self.bbox_alpha = bbox_alpha
        self.mask_alpha = mask_alpha
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

    def lincomb_mask_loss(self, gt_loc_all, gt_boxes, gt_masks, gt_labels, batch_positive_idx, batch_best_truth_idx, pred_mask_proto, pred_masks):
        """ 计算mask损失
        :param gt_loc_all: [batch, num_priors, 4(cx,cy,cw,ch)]
        :param gt_boxes: [batch, num_boxes, 4(x1,y1,x2,y2)]
        :param gt_masks: [batch, h, w, num_objs]
        :param gt_labels: [batch, num_objs]
        :param batch_positive_idx: [batch, num_priors]
        :param batch_best_truth_idx: [batch, num_priors]
        :param pred_mask_proto: [batch, 1/4, 1/4, 32]
        :param pred_masks: [batch, num_priors, 32]
        :return:
        """
        mask_h, mask_w = pred_mask_proto.shape[1:3]

        for i in range(self.batch_size):
            # [h,w,num_objs] => [1/4, 1/4, num_objs] => [num_objs, 1/4, 1/4]
            cur_gt_masks_resize = tf.image.resize(gt_masks[i], size=(mask_h, mask_w), method=tf.image.ResizeMethod.BILINEAR)
            cur_gt_masks_resize = np.array(cur_gt_masks_resize > 0.5, dtype=np.float32).transpose([2,0,1])

            # 正样本(conf>0)的索引
            pos_idx = batch_positive_idx[i]
            # 正样本边框(conf>0)的索引, 每个prior对应一个边框
            best_truth_pos_idx = batch_best_truth_idx[i][pos_idx]

            # pos_gt_box = pos_gt_decode(loc_all[i], self.priors)[pos_idx]
            cur_pos_gt_box = gt_boxes[i][pos_idx]

            if pos_idx.size(0) == 0:
                continue

            # [1/4, 1/4, 32]
            cur_pred_mask_proto = pred_mask_proto[i]
            # [pos_nums, 32]
            cur_pos_mask_coef = pred_masks[i][pos_idx]

            if cur_pos_mask_coef.shape[0] > self.masks_to_train:
                select = np.random.permutation(cur_pos_mask_coef.shape[0])[:self.masks_to_train]
                cur_pos_mask_coef = cur_pos_mask_coef[select]
                pos_idx = pos_idx[select]
                cur_pos_gt_box = cur_pos_gt_box[select]

            # [n, 1/4, 1/4]
            cur_pos_gt_masks_resize = cur_gt_masks_resize[pos_idx]
            # [n]
            cur_pos_gt_lables = gt_labels[i][pos_idx]


    def lincomb_mask_loss(self, pos, idx_t, mask_data, proto_data, masks, gt_box_t):
        """ 计算mask损失

        :param pos: 正样本mask索引, [batch, num_priors]
        :param idx_t: 每个prior对应的目标边框索引, [batch, num_priors]
        :param mask_data: [batch, num_priors, 32]
        :param proto_data: [batch, h/4, w/4, 32]
        :param masks: [batch, im_h, im_w, num_objs]
        :param gt_box_t: [batch, num_objs, 4(x1,y1,x2,y2)]
        :return:
        """
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        loss_m = 0
        for idx in range(mask_data.size(0)):
            # [im_h, im_w, num_objs] => [1/4, 1/4, num_objs]
            downsampled_masks = tf.image.resize(masks[idx], size=(mask_h, mask_w), method=tf.image.ResizeMethod.BILINEAR)
            downsampled_masks = np.array(downsampled_masks > 0.5, dtype=np.float32)
            # downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
            #                                   mode=interpolation_mode, align_corners=False).squeeze(0)
            # downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

            # if cfg.mask_proto_binarize_downsampled_gt:
            #     downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef = mask_data[idx, cur_pos, :]

            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            if old_num_pos > self.masks_to_train:
                select = np.random.permutation(proto_coef.shape[0])[:self.masks_to_train]
                proto_coef = proto_coef[select]
                pos_idx_t = pos_idx_t[select]

                # perm = torch.randpserm(proto_coef.size(0))
                # select = perm[:cfg.masks_to_train]
                # proto_coef = proto_coef[select, :]
                # pos_idx_t  = pos_idx_t[select]

                pos_gt_box_t = pos_gt_box_t[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]

            # Size: [h, w, 32] x [pos_nums, 32].T = [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.transpose()
            pred_masks = tf.sigmoid(pred_masks)

            pre_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(mask_t, np.clip(pred_masks,0,1))
            # pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')

            pos_gt_csize = np.concatenate(((pos_gt_box_t[,:2] + pos_gt_box_t[,2:])/2.,
                                           pos_gt_box_t[,2:] - pos_gt_box_t[,:2]), axis=1)
            gt_box_width  = pos_gt_csize[:, 2] * mask_w
            gt_box_height = pos_gt_csize[:, 3] * mask_h
            pre_loss = np.sum(pre_loss) / gt_box_width / gt_box_height

            # if cfg.mask_proto_normalize_emulate_roi_pooling:
            #     weight = mask_h * mask_w if cfg.mask_proto_crop else 1
            #     pos_gt_csize = center_size(pos_gt_box_t)
            #     gt_box_width  = pos_gt_csize[:, 2] * mask_w
            #     gt_box_height = pos_gt_csize[:, 3] * mask_h
            #     pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight

            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            loss_m += np.sum(pre_loss)

        losses = loss_m * self.mask_alpha / mask_h / mask_w
        return losses

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


