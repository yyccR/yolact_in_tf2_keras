import numpy as np
import tensorflow as tf


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


def nms(boxes, masks, scores, image_size=640, iou_threshold=0.5, conf_thresh=0.05):
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

    from utils.cython_nms import nms as cnms

    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * image_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > conf_thresh
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, iou_threshold)
        keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    classes = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:cfg.max_num_detections]
    scores = scores[:cfg.max_num_detections]

    idx = idx[idx2]
    classes = classes[idx2]

    # Undo the multiplication above
    return boxes[idx] / cfg.max_size, masks[idx], classes, scores


def yolact_detect(conf_preds, decoded_boxes, mask_data, inst_data, batch_size):
    conf_preds = tf.transpose(conf_preds, perm=[0, 2, 1])
    nms(
        boxes=decoded_boxes,
        masks=mask_data,
        scores=conf_preds,
        # image_size=
    )


class YolactDetect:
    """yolact detect层, box decode, nms操作"""
    def __init__(self, image_size, num_classes, batch_size, bkg_label, top_k, conf_thresh, nms_thresh):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.bkg_label = bkg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh


    def __call__(self, inputs, *args, **kwargs):
        boxes, masks, scores, priors = inputs
        # [batch, -1, 4(x1,y1,x2,y2)]
        decoded_boxes = decode(loc=boxes, priors=priors)
