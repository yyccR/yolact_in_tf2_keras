import cv2
import numpy as np
import tensorflow as tf
from box_utils import iou, decode, match, crop


class MultiBoxLoss:
    def __init__(self,
                 batch_size, num_classes, priors, conf_alpha=1, bbox_alpha=1.5,
                 semantic_segmentation_alpha=1, mask_alpha=6.125, pos_thresh=0.4,
                 neg_thresh=0.3, masks_to_train=100, ohem_negpos_ratio=3):
        self.batch_size = batch_size
        self.num_classes = num_classes
        # [num_priors, 4]
        self.priors = priors
        self.conf_alpha = conf_alpha
        self.bbox_alpha = bbox_alpha
        self.mask_alpha = mask_alpha
        self.semantic_segmentation_alpha = semantic_segmentation_alpha
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.masks_to_train = masks_to_train
        self.ohem_negpos_ratio = ohem_negpos_ratio

    def smooth_l1_loss(self, loc_p, loc_t):
        """计算目标边框损失"""
        # print(loc_p.shape, loc_t.shape)
        diff = tf.abs(loc_t - loc_p)
        # less_than_one = tf.cast(tf.keras.backend.less(diff, 1.0), dtype=tf.float32)
        less_than_one = tf.cast(diff < 1.0, dtype=tf.float32)
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        loss = tf.reduce_sum(loss)
        return loss

    def ohem_conf_loss(self, pred_cls, true_cls, pos, batch_size):
        """ 计算分类损失
        :param pred_cls: [batch, num_priors, num_classes]
        :param true_cls: [batch, num_priors]
        :param pos: [batch, num_priors]
        :param batch_size:
        :return:
        """
        # Compute max conf across batch for hard negative mining
        # batch_conf = np.reshape(conf_data, [-1, self.num_classes])
        # conf_max = np.max(batch_conf)
        # loss_c = np.log(np.sum(np.exp(batch_conf-conf_max), 1)) + conf_max - batch_conf[:, 0]
        # loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]
        # batch_conf = tf.math.softmax(batch_conf, dim=1)
        # loss_c = np.max(batch_conf[:, 1:], axis=1)
        # loss_c = np.max(batch_conf, axis=1)
        loss_c = tf.reduce_max(pred_cls, axis=-1)

        # Hard Negative Mining
        # loss_c = np.reshape(loss_c, [batch_size, -1])
        # 过滤掉正样本
        pos_indices = tf.where(pos)
        pos_samples = tf.zeros((pos_indices.get_shape()[0]))
        loss_c = tf.tensor_scatter_nd_update(loss_c, pos_indices, pos_samples)
        # loss_c[pos] = 0
        # 过滤掉容易分的负样本
        easy_neg_indices = tf.where(true_cls < 0)
        easy_neg_samples = tf.zeros((easy_neg_indices.get_shape()[0]))
        loss_c = tf.tensor_scatter_nd_update(loss_c, easy_neg_indices, easy_neg_samples)
        # loss_c[conf_t < 0] = 0
        # loss_c = -np.sort(-loss_c, axis=1)

        # 从大到小排序, 拿到下标索引
        loss_idx = tf.argsort(loss_c, axis=1, direction='DESCENDING')
        # loss_idx = np.argsort(-loss_c, axis=1)
        # _, loss_idx = loss_c.sort(1, descending=True)
        # 从小到大排序, 拿到[0~num_priors]每个索引对应的loss排序值,比如第一位loss排在18，第二位排到了300,
        # 那么如果只拿前100个负样本计算损失的话，第二位的排名大于100不会被采用
        # idx_rank = np.argsort(loss_idx, axis=1)
        idx_rank = tf.argsort(loss_idx, axis=1, direction='ASCENDING')
        # _, idx_rank = loss_idx.sort(1)
        # num_pos = pos.long().sum(1, keepdim=True)
        # num_pos = np.sum(pos, axis=1)
        num_pos = tf.reduce_sum(tf.cast(pos, dtype=tf.int32), axis=1)

        # num_neg = torch.clamp(self.ohem_negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 计算负样本个数, 这里负样本为正样本3倍,并且不超过原样本总数
        # num_neg = np.minimum(self.ohem_negpos_ratio * num_pos, pos.shape[1] - 1)
        num_neg = tf.minimum(self.ohem_negpos_ratio * num_pos, pos.shape[1] - 1)
        # 先对loss_c从大到小排序, 得到排序下标索引, 再对这个排序下标索引从小到大排序拿到索引
        # neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = idx_rank < tf.broadcast_to(num_neg[:, None], idx_rank.get_shape())
        neg = tf.cast(neg, dtype=tf.int32)
        # neg = idx_rank < np.broadcast_to(num_neg[:, None], idx_rank.shape)

        # 去掉那些正样本和容易分的负样本 pos为conf_t>0的，还有conf_t=0和conf_t=-1的，-1的是容易分的负样本
        neg = tf.tensor_scatter_nd_update(neg, pos_indices, tf.cast(pos_samples, dtype=tf.int32))
        # neg[pos] = 0
        neg = tf.tensor_scatter_nd_update(neg, easy_neg_indices, tf.cast(easy_neg_samples,dtype=tf.int32))
        # neg[conf_t < 0] = 0

        # [batch, num_priors] => [batch, num_priors, 1] => [batch, num_priors, num_classes]
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # pos_idx = np.broadcast_to(np.expand_dims(pos, axis=-1), conf_data.shape)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # neg_idx = np.broadcast_to(np.expand_dims(neg, axis=-1), conf_data.shape)

        # conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        pos_neg_indices = (pos+neg) > 0
        # conf_p = tf.reshape(conf_data[pos_neg_indices], (-1, self.num_classes))
        # print(pred_cls.shape,true_cls.shape, pos_neg_indices.shape, pos.shape)
        conf_p = tf.gather_nd(pred_cls,tf.where(pos_neg_indices))
        # targets_weighted = conf_t[(pos + neg).gt(0)]
        conf_t = true_cls[pos_neg_indices.numpy()]
        loss_c = tf.keras.losses.SparseCategoricalCrossentropy()(conf_t, conf_p)
        # loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')
        # loss_c = loss_c.sum()

        return self.conf_alpha * loss_c

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, mask_h, mask_w, num_classes = segment_data.shape()
        loss_s = 0

        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            # downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
            #                                   mode=interpolation_mode, align_corners=False).squeeze(0)
            # downsampled_masks = downsampled_masks.gt(0.5).float()
            downsampled_masks = tf.image.resize(mask_t[idx], size=(mask_h, mask_w),
                                                method=tf.image.ResizeMethod.BILINEAR)
            downsampled_masks = np.array(downsampled_masks > 0.5, dtype=np.float32)

            # Construct Semantic Segmentation
            segment_t = np.zeros_like(cur_segment)
            for obj_idx in range(downsampled_masks.shape[0]):
                segment_t[cur_class_t[obj_idx]] = np.max(segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx])

            # loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')
            loss_s = tf.keras.losses.BinaryCrossentropy(from_logits=True)(segment_t, cur_segment)

        return self.semantic_segmentation_alpha * loss_s / mask_h / mask_w

    def lincomb_mask_loss(self, pos, idx_t, pred_mask, pred_mask_proto, gt_masks, prior_gt_boxes):
        """ 计算mask损失

        :param pos: 正样本mask索引, [batch, num_priors]
        :param idx_t: 每个prior对应的目标边框索引, [batch, num_priors]
        :param pred_mask: [batch, num_priors, 32]
        :param pred_mask_proto: [batch, h/4, w/4, 32]
        :param gt_masks: [batch, im_h, im_w, num_objs]
        :param prior_gt_boxes: [batch, num_priors, 4(x1,y1,x2,y2)]
        :return:
        """
        mask_h = pred_mask_proto.shape[1]
        mask_w = pred_mask_proto.shape[2]

        loss_m = 0
        for idx in range(pred_mask.shape[0]):
            # [im_h, im_w, num_objs] => [1/4, 1/4, num_objs]
            downsampled_masks = tf.image.resize(gt_masks[idx], size=(mask_h, mask_w),
                                                method=tf.image.ResizeMethod.BILINEAR)
            downsampled_masks = np.array(downsampled_masks > 0.5, dtype=np.float32)
            # downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
            #                                   mode=interpolation_mode, align_corners=False).squeeze(0)
            # downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

            # if cfg.mask_proto_binarize_downsampled_gt:
            #     downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx][cur_pos]
            pos_gt_box_t = prior_gt_boxes[idx][cur_pos]

            if pos_idx_t.shape[0] == 0:
                continue
            proto_masks = pred_mask_proto[idx]
            proto_coef = pred_mask[idx][cur_pos]

            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.shape[0]
            if old_num_pos > self.masks_to_train:
                select = tf.random.shuffle(tf.range(proto_coef.shape[0]))[:self.masks_to_train, ]
                # select = np.random.permutation(proto_coef.shape[0])[:self.masks_to_train]
                proto_coef = tf.gather(proto_coef,select)
                # proto_coef = proto_coef[select]
                # pos_idx_t = tf.gather(pos_idx_t,select)
                pos_idx_t = pos_idx_t[select]

                # perm = torch.randpserm(proto_coef.size(0))
                # select = perm[:cfg.masks_to_train]
                # proto_coef = proto_coef[select, :]
                # pos_idx_t  = pos_idx_t[select]

                pos_gt_box_t = tf.gather(pos_gt_box_t,select)
                # pos_gt_box_t = pos_gt_box_t[select]

            # num_pos = proto_coef.shape[0]
            # [num_selects, mask_h, mask_w] => [mask_h, mask_w, num_selects]
            mask_t = downsampled_masks[:, :, pos_idx_t]

            # Size: [h, w, 32] x [pos_nums, 32].T = [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ tf.transpose(proto_coef, [1, 0])
            pred_masks = tf.sigmoid(pred_masks)
            pred_masks = crop(pred_masks, pos_gt_box_t)
            # pre_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(mask_t, pred_masks)
            pre_loss = tf.keras.losses.BinaryCrossentropy()(mask_t, pred_masks)
            # pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')

            # pos_gt_csize = np.concatenate(((pos_gt_box_t[..., :2] + pos_gt_box_t[..., 2:]) / 2.,
            #                                pos_gt_box_t[..., 2:] - pos_gt_box_t[..., :2]), axis=1)
            # gt_box_width = pos_gt_csize[..., 2] * mask_w
            # gt_box_height = pos_gt_csize[..., 3] * mask_h

            # gt_box_width = (pos_gt_box_t[..., 2:] - pos_gt_box_t[..., :2]) * mask_w
            # gt_box_height = (pos_gt_box_t[..., 2:] - pos_gt_box_t[..., :2]) * mask_h
            # pre_loss = tf.reduce_sum(pre_loss) / gt_box_width / gt_box_height

            # if cfg.mask_proto_normalize_emulate_roi_pooling:
            #     weight = mask_h * mask_w if cfg.mask_proto_crop else 1
            #     pos_gt_csize = center_size(pos_gt_box_t)
            #     gt_box_width  = pos_gt_csize[:, 2] * mask_w
            #     gt_box_height = pos_gt_csize[:, 3] * mask_h
            #     pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight

            # If the number of masks were limited scale the loss accordingly
            # if old_num_pos > num_pos:
            #     pre_loss *= old_num_pos / num_pos
            #
            # loss_m += tf.reduce_sum(pre_loss)
            loss_m += pre_loss

        # losses = loss_m * self.mask_alpha / mask_h / mask_w
        losses = loss_m * self.mask_alpha
        return losses

    def __call__(self, inputs, gt_boxes, gt_labels, gt_masks, *args, **kwargs):
        """
        :param inputs: [box, cls, mask, mask_proto, semantic_seg_conv]
        :param gt_boxes: [batch, n, gt_boxes]
        :param gt_labels: [batch, n]
        :param gt_masks:
        :param args:
        :param kwargs:
        :return:
        """
        # pred_box: [batch, num_priors, 4(cx,cy,cw,ch)]
        # pred_cls: [batch, num_priors, cls_nums]
        # pred_mask: [batch, num_priors, mask_proto_channels]
        # pred_mask_proto: [batch, 1/4, 1/4, 32]
        # pred_semantic_seg: [batch, 1/8, 1/8, classes-1]
        pred_box, pred_cls, pred_mask, pred_mask_proto = inputs
        # pred_box = pred_box.numpy()
        # pred_cls = pred_cls.numpy()
        # pred_mask = pred_mask.numpy()
        # pred_mask_proto = pred_mask_proto.numpy()

        batch_loc = []
        batch_conf = []
        batch_prior_gt_boxes = []
        batch_best_truth_idx = []
        for i in range(self.batch_size):
            # 这里给每个prior找到对应的边框, overlap太小的不要, 类似于faster-rcnn/mask-rcnn里的anchors做法
            # loc:[num_priors, 4(cx,cy,cw,ch)]
            # conf:[num_priors]
            # best_truth_idx:[num_priors]
            loc, prior_gt_boxes, conf, best_truth_idx = match(
                self.priors, gt_boxes[i], gt_labels[i], self.pos_thresh, self.neg_thresh)
            # prior_gt_boxes = gt_boxes[i][best_truth_idx]

            batch_loc.append(loc)
            batch_conf.append(conf)
            batch_prior_gt_boxes.append(prior_gt_boxes)
            batch_best_truth_idx.append(best_truth_idx)

        loc_all = np.array(batch_loc, dtype=np.float32)
        conf_all = np.array(batch_conf, dtype=np.float32)
        prior_gt_boxes_all = np.array(batch_prior_gt_boxes, dtype=np.float32)
        batch_best_truth_idx = np.array(batch_best_truth_idx, dtype=np.int32)

        # 所有非背景边框损失
        pos = conf_all > 0
        box_loss = self.smooth_l1_loss(pred_box[pos], loc_all[pos])
        box_loss /= self.batch_size

        # mask损失
        mask_loss = self.lincomb_mask_loss(pos, batch_best_truth_idx, pred_mask, pred_mask_proto, gt_masks,
                                           prior_gt_boxes_all)
        mask_loss /= self.batch_size

        # 类别损失
        cls_loss = self.ohem_conf_loss(pred_cls, conf_all, pos, self.batch_size)
        cls_loss /= self.batch_size

        # 语义分割损失
        # sem_seg_loss = self.semantic_segmentation_loss(pred_semantic_seg, gt_masks, gt_labels)
        # sem_seg_loss /= self.batch_size

        # total_loss = (box_loss + mask_loss + cls_loss + sem_seg_loss)
        total_loss = (box_loss + mask_loss + cls_loss)

        # return box_loss, mask_loss, cls_loss, sem_seg_loss, total_loss
        return box_loss, mask_loss, cls_loss, total_loss
