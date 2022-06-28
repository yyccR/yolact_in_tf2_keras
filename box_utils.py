
import numpy as np
import tensorflow as tf


def iou(box_a, box_b, eps=1e-7):
    """
    :param box_a: [n, 4(x1,y1,x2,y2)]
    :param box_b: [m, 4(x1,y1,x2,y2)]
    :return: [n,m] or [batch, n, m]
    """

    box_a = box_a[..., None, :]
    # [n, m]
    inter = (np.minimum(box_a[..., 3], box_b[...,3]) - np.maximum(box_a[...,1], box_b[...,1])) * \
            (np.minimum(box_a[..., 2], box_b[...,2]) - np.maximum(box_a[...,0], box_b[...,0]))

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])
    # [n, m]
    union = area_a + area_b

    iou = inter / (union + eps)
    return iou


def encode(matched, priors):
    """ 计算真实边框与先验框之间的平移缩放量
    :param matched: 真实边框: [n, 4(x1,y1,x2,y2)]
    :param priors: 先验框: [n, 4(x,y,w,h)]
    :return: [n, 4(cx,cy,cw,ch)]
    """
    variances = [0.1, 0.2]

    # 中心点平移量
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    # 计算缩放量, 类似于yolov3,faster-rcnn/mask-rcnn
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    loc = np.concatenate([g_cxcy, g_wh],axis=1)
    return loc


def decode(loc, priors):
    """ 将预测偏移缩放量映射到prior上, 处理成[x,y,w,h]

    :param loc: shape [batch,-1,4]
    :param priors: shape [-1, 4]
    :return: [batch,-1,4(x,y,x,y)]
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


def match(priors, ground_true_boxes, ground_true_labels, pos_thresh=0.5, neg_thresh=0.4):
    """ 这里给每个prior找到对应的边框, overlap太小的不要, 类似于faster-rcnn/mask-rcnn里的anchors做法
    :param priors: [num_priors, 4(x,y,w,h)]
    :param ground_true_boxes: [num_boxes, 4(x1,y1,x2,y2)]
    :param ground_true_labels: [num_boxes]
    :param pos_thresh: 正样本阈值
    :param neg_thresh: 负样本阈值
    :return: loc:[num_prioris, 4(cx,cy,cw,ch)], conf:[num_prioris], best_truth_idx:[num_prioris]
    """

    priors_xyxy = np.concatenate((priors[..., :2] - priors[..., 2:] / 2, priors[..., :2] + priors[..., 2:] / 2), 1)
    # [num_boxes, num_priors]
    overlap = iou(ground_true_boxes, priors_xyxy)
    # 每个prior对应的真实边框的最大overlap,以及对应的真实边框的id
    best_truth_overlap = np.max(overlap, axis=0)
    best_truth_idx = np.argmax(overlap,axis=0)

    for _ in range(overlap.shape[0]):
        # 找到每个真实边框与prior的最大overlap, 及其id
        best_prior_overlap = np.max(overlap, axis=1)
        best_prior_idx = np.argmax(overlap, axis=1)

        # 在上面最大overlap列表里找到最大的一个,
        j = best_prior_overlap.argmax(0)
        # 最大的那个对应的prior索引
        i = best_prior_idx[j]

        # 既然知道了哪个最大，就标记为-1表明不再参与下次筛选
        overlap[:, i] = -1
        overlap[j, :] = -1

        # 这里标记prior对应overlap为2，避免太小被筛选掉
        best_truth_overlap[i] = 2
        best_truth_idx[i] = j

    # 这里最终每个prior都会找到一个对应的真实边框
    matches = ground_true_boxes[best_truth_idx]
    # 这里模型默认第一位预测的是背景, 所以所有标签顺延+1
    conf = ground_true_labels[best_truth_idx] + 1

    # (negative 样本)0.4 < (native 样本) < 0.5(positive 样本)
    conf[best_truth_overlap < pos_thresh] = -1
    conf[best_truth_overlap > neg_thresh] = 0

    # 计算真实边框与先验框之间的平移缩放量
    loc = encode(matches, priors)
    return loc, conf, best_truth_idx
