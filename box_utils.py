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
    inter = (np.minimum(box_a[..., 3], box_b[..., 3]) - np.maximum(box_a[..., 1], box_b[..., 1])) * \
            (np.minimum(box_a[..., 2], box_b[..., 2]) - np.maximum(box_a[..., 0], box_b[..., 0]))

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
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    # 计算缩放量, 类似于yolov3,faster-rcnn/mask-rcnn
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    loc = np.concatenate([g_cxcy, g_wh], axis=1)
    return loc


def decode(loc, priors):
    """ 将预测偏移缩放量映射到prior上, 处理成[x,y,w,h]

    :param loc: shape [batch,num_priors,4]
    :param priors: shape [num_priors, 4]
    :return: [batch,num_priors,4(x,y,x,y)]
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


def nms(decoded_boxes, masks, scores, image_size=640, iou_threshold=0.5, conf_thresh=0.25, top_k=200):
    """

    :param decoded_boxes: [batch, num_priors, 4(x1,y1,x2,y2)]
    :param masks: [batch, num_priors, mask_dim=32]
    :param scores: [batch, num_priors, num_classes+1]
    :param image_size:
    :param iou_threshold:
    :param conf_thresh:
    :param top_k:
    :return:
    """
    batch_size = decoded_boxes.shape[0]
    num_classes = scores.shape[2]
    decoded_boxes = decoded_boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()

    batch_boxes = []
    batch_masks = []
    batch_scores = []
    batch_classes = []
    for i in range(batch_size):
        # cur_socres = scores[i, :, 1:]
        cur_socres = scores[i]
        max_socres = np.max(cur_socres, axis=-1)
        keep = max_socres > conf_thresh

        cur_boxes = decoded_boxes[i][keep]
        cur_boxes = cur_boxes * image_size
        cur_masks = masks[i][keep]
        cur_socres = cur_socres[keep]

        cur_nms_boxes = []
        cur_nms_masks = []
        cur_nms_scores = []
        cur_nms_classes = []
        for _cls in range(num_classes):
            cls_scores = cur_socres[:, _cls]
            conf_mask = cls_scores > conf_thresh
            # idx = np.arange(cls_scores.size(0))

            cls_scores = cls_scores[conf_mask]
            cls_boxes = cur_boxes[conf_mask]
            cls_mask = cur_masks[conf_mask]
            # idx = idx[conf_mask]

            if cls_scores.shape[0] == 0:
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
            cur_nms_classes.append(nms_keep * 0 + _cls)
            cur_nms_masks.append(cls_mask[nms_keep])
            # idx_lst.append(idx[keep])
            # cls_lst.append(keep * 0 + _cls)
            # scr_lst.append(cls_scores[keep])

        # idx = torch.cat(idx_lst, dim=0)
        # classes = torch.cat(cls_lst, dim=0)
        # scores = torch.cat(scr_lst, dim=0)
        cur_nms_boxes = np.concatenate(cur_nms_boxes, axis=0)
        cur_nms_masks = np.concatenate(cur_nms_masks, axis=0)
        cur_nms_scores = np.concatenate(cur_nms_scores, axis=0)
        cur_nms_classes = np.concatenate(cur_nms_classes, axis=0)

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
    best_truth_idx = np.argmax(overlap, axis=0)

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
    # conf = ground_true_labels[best_truth_idx] + 1
    conf = ground_true_labels[best_truth_idx]

    # (negative 样本)0.4 < (native 样本) < 0.5(positive 样本)
    conf[best_truth_overlap < pos_thresh] = -1
    conf[best_truth_overlap < neg_thresh] = 0

    # 计算真实边框与先验框之间的平移缩放量
    loc = encode(matches, priors)
    return loc, conf, best_truth_idx


def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.maximum(x1 - padding, 0)
    x2 = np.minimum(x2 + padding, img_size)

    return x1, x2


def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = np.tile(np.arange(w).reshape([1, -1, 1]), (h, 1, n))
    cols = np.tile(np.arange(h).reshape([-1, 1, 1]), (1, w, n))

    masks_left = rows >= x1.reshape([1, 1, -1])
    masks_right = rows < x2.reshape([1, 1, -1])
    masks_up = cols >= y1.reshape([1, 1, -1])
    masks_down = cols < y2.reshape([1, 1, -1])

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask


def detect(pred_boxes, pred_classes, pred_masks, pred_proto, image_size=640, iou_threshold=0.5, conf_thresh=0.25,
           top_k=200):
    """ 输出最终格式边框,类别,masks

    :param pred_boxes: 已经decode好的box, 训练和推理的时候不一样, [batch, num_priors, 4(x1,y1,x2,y2]
    :param pred_classes: [batch, num_priors, num_classes]
    :param pred_masks: [batch, num_priors, mask_dim]
    :param pred_proto: [batch, h/4, w/4, mask_dim]
    :param image_size:
    :param iou_threshold:
    :param conf_thresh:
    :param top_k:
    :return: out_boxes: [batch, n, 4(x1,y1,x2,y2)] ∈ [0,image_size]
             out_classes: [batch, n]
             out_scores: [batch, n]
             out_masks: [batch, image_h, image_w, n]
    """
    # 对每个batch的数据做非极大抑制, box处理到[0,image_size]
    nms_boxes, nms_masks, nms_classes, nms_scores = nms(
        decoded_boxes=pred_boxes,
        masks=pred_masks,
        scores=pred_classes,
        image_size=image_size,
        iou_threshold=iou_threshold,
        conf_thresh=conf_thresh,
        top_k=top_k
    )

    # 没有数据返回空
    if not nms_classes or len(nms_classes[0]) <= 0:
        return

    out_masks = []
    out_boxes = []
    out_classes = []
    out_scores = []
    for i in range(len(nms_classes)):
        # [num_detections, mask_dim] => [mask_dim, num_detections]
        cur_masks = np.transpose(nms_masks[i], [1, 0])
        # [h/4, w/4, mask_dim] @ [mask_dim, num_detections] => [h/4, w/4, num_detections]
        masks = pred_proto[i] @ cur_masks
        # 只保留每个目标边框里面的mask
        masks = crop(masks, nms_boxes[i])
        # mask处理到输入图片的大小
        up_sample_masks = tf.image.resize(masks, size=(image_size, image_size), method=tf.image.ResizeMethod.BILINEAR)
        up_sample_masks = np.array(up_sample_masks > 0.5,dtype=np.int32)

        # 目标边框处理到输入图片大小
        boxes_x1, boxes_y1 = sanitize_coordinates(nms_boxes[i][:, 0], nms_boxes[i][:, 2], image_size)
        boxes_x2, boxes_y2 = sanitize_coordinates(nms_boxes[i][:, 1], nms_boxes[i][:, 3], image_size)
        nms_boxes_x1y1x2y2 = np.concatenate(
            [boxes_x1[:,None],boxes_y1[:,None],boxes_x2[:,None],boxes_y2[:,None]],axis=-1)

        # 顶部k个
        top_k_id = np.argsort(-nms_scores[i])[:top_k]
        # if len(top_k_id) == 1:
        #     top_k_id = np.array([top_k_id])
        out_boxes.append(nms_boxes_x1y1x2y2[top_k_id])
        out_classes.append(nms_classes[i][top_k_id])
        out_scores.append(nms_scores[i][top_k_id])
        out_masks.append(up_sample_masks[:,:,top_k_id])

    return out_boxes, out_classes, out_scores, out_masks