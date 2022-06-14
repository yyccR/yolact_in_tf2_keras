import numpy as np
import tensorflow as tf

# import pyximport
# pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)


def decode(loc, priors):
    """ 将预测偏移缩放量和anchors 处理成[x,y,w,h]

    :param loc: shape [batch,-1,4]
    :param priors: shape [1, -1, 4]
    :return: [batch,-1,4(x,y,w,h)]
    """
    variances = [0.1, 0.2]

    boxes_xy = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
    boxes_wh = priors[..., 2:] * tf.math.exp(loc[..., 2:] * variances[1])

    boxes_x1y1 = boxes_xy - boxes_wh / 2.
    boxex_x2y2 = boxes_x1y1 + boxes_wh

    boxes = tf.keras.layers.Concatenate(axis=-1)(
        [boxes_x1y1, boxex_x2y2]
    )
    return boxes


def nms(decoded_boxes, masks, scores, image_size=640, iou_threshold=0.5, conf_thresh=0.05, top_k=200):
    """

    :param decoded_boxes: [batch, num_prior, 4(x1,y1,x2,y2)]
    :param masks: [batch, num_prior, mask_dim=32]
    :param scores: [batch, num_prior,num_classes+1]
    :param image_size:
    :param iou_threshold:
    :param conf_thresh:
    :param top_k:
    :return:
    """
    batch_size = decoded_boxes.shape[0]

    batch_boxes = []
    batch_masks = []
    batch_scores = []
    batch_classes = []
    for i in range(batch_size):
        cur_socres = scores[i, :, 1:]
        cur_socres = np.max(cur_socres, axis=-1)
        keep = cur_socres > conf_thresh

        cur_boxes = decoded_boxes[i][keep]
        cur_boxes = cur_boxes * image_size
        cur_masks = masks[i][keep]
        cur_socres = cur_socres[keep]

        cur_nms_boxes = []
        cur_nms_masks = []
        cur_nms_scores = []
        cur_nms_classes = []
        num_classes = cur_socres.shape[1]
        for _cls in range(num_classes):
            cls_scores = cur_socres[:, _cls]
            conf_mask = cls_scores > conf_thresh
            # idx = np.arange(cls_scores.size(0))

            cls_scores = cls_scores[conf_mask]
            cls_boxes = cur_boxes[conf_mask]
            cls_mask = cur_masks[conf_mask]
            # idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue

            # preds = np.concatenate([cls_boxes, cls_scores[:, None]], axis=1)
            # nms_keep = cnms(preds, iou_threshold)
            nms_keep = tf.image.non_max_suppression(
                boxes=cls_boxes,
                scores=cls_scores,
                max_output_size=top_k,
                iou_threshold=iou_threshold
            ).numpy()

            cur_nms_boxes.append(cls_boxes[nms_keep])
            cur_nms_scores.append(cls_scores[nms_keep])
            cur_nms_classes.append([keep] * 0 + _cls)
            cur_nms_masks.append(cls_mask[keep])
            # idx_lst.append(idx[keep])
            # cls_lst.append(keep * 0 + _cls)
            # scr_lst.append(cls_scores[keep])

        # idx = torch.cat(idx_lst, dim=0)
        # classes = torch.cat(cls_lst, dim=0)
        # scores = torch.cat(scr_lst, dim=0)
        cur_nms_boxes = np.concatenate(cur_nms_boxes,axis=0)
        cur_nms_masks = np.concatenate(cur_nms_masks,axis=0)
        cur_nms_scores = np.concatenate(cur_nms_scores,axis=0)
        cur_nms_classes = np.concatenate(cur_nms_classes,axis=0)

        sorted_idx = cur_nms_scores.argsort(axis=0)[::-1][:top_k]
        batch_boxes.append(cur_nms_boxes[sorted_idx] / image_size)
        batch_masks.append(cur_nms_masks[sorted_idx])
        batch_scores.append(cur_nms_scores[sorted_idx])
        batch_classes.append(cur_nms_classes[sorted_idx])

    # Undo the multiplication above
    # return boxes[idx] / cfg.max_size, masks[idx], classes, scores
    return batch_boxes, batch_masks, batch_classes, batch_scores


# class YolactDetect:
#     """yolact detect层, box decode, nms操作"""
#     def __init__(self,
#                  image_size,
#                  num_classes,
#                  batch_size,
#                  bkg_label,
#                  top_k=200,
#                  conf_thresh=0.05,
#                  nms_thresh=0.5):
#         self.image_size = image_size
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.bkg_label = bkg_label
#         self.top_k = top_k
#         self.conf_thresh = conf_thresh
#         self.nms_thresh = nms_thresh
#
#     def __call__(self, inputs, *args, **kwargs):
#         boxes, masks, scores, priors = inputs
#         # [batch, -1, 4(x1,y1,x2,y2)]
#         decoded_boxes = self.decode(loc=boxes, priors=priors)
#
#     def decode(self, loc, priors):
#         """ 将预测偏移缩放量和anchors 处理成[x1,y1,x2,y2]
#
#         :param loc: shape [batch,-1,4]
#         :param priors: shape [1, -1, 4]
#         :return: [batch,-1,4(x1,y1,x2,y2)]
#         """
#         variances = [0.1, 0.2]
#
#         boxes_xy = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
#         boxes_wh = priors[..., 2:] * tf.math.exp(loc[..., 2:] * variances[1])
#
#         boxes_x1y1 = boxes_xy - boxes_wh / 2.
#         boxex_x2y2 = boxes_x1y1 + boxes_wh
#
#         boxes = tf.keras.layers.Concatenate(axis=-1)(
#             [boxes_x1y1, boxex_x2y2]
#         )
#         return boxes
#
#     def detect(self, decoded_boxes, pred_mask, pred_cls):
#         """
#
#         :param decoded_boxes: [num_prior, 4(x1,y1,x2,y2)]
#         :param pred_mask: [num_prior, mask_dim=32]
#         :param pred_cls: [num_prior, num_classes]
#         :return:
#         """
#         cur_socres = pred_cls[..., 1:]
#         # 概率最大的类别
#         cur_socres = tf.reduce_max(cur_socres, axis=-1)
#         keep = cur_socres > self.conf_thresh
#
#         decoded_boxes = decoded_boxes[keep]
#         pred_mask = pred_mask[pred_mask]
#         scores = cur_socres[keep]
#

