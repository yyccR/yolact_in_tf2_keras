import os

import cv2
import tqdm
import numpy as np
import pandas as pd
from yolact import Yolact
from data.generate_coco_data import CoCoDataGenrator
from box_utils import decode, detect

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def ap_per_class(box_correct, mask_correct, pred_conf, pred_cls, target_cls, eps=1e-16):
    """ 计算每个类别的ap
    :param box_correct: [m, 10], 记录着每个预测边框在对应的iou阈值下是否有匹配的真实目标边框
    :param mask_correct: [m, 10], 记录着每个预测mask在对应的iou阈值下是否有匹配的真实目标mask
    :param pred_conf: [m]
    :param pred_cls: [m]
    :param target_cls: [n]
    :return: [box_metric:(unique_classes, ap, precision, recall),
             mask_metric:(unique_classes, ap, precision, recall)]
    """
    # 逆序从大到小
    i = np.argsort(-pred_conf)
    b_correct, m_correct, pred_conf, pred_cls = box_correct[i], mask_correct[i], pred_conf[i], pred_cls[i]

    # 去重类别
    unique_classes, num_per_classes = np.unique(target_cls, return_counts=True)
    num_classes = unique_classes.shape[0]

    # box_metric and mask_metric
    metrics = []
    for correct in [b_correct, m_correct]:
        # 分别计算box和mask的mAP
        ap = np.zeros((num_classes, correct.shape[1]))
        precision = np.zeros((num_classes))
        recall = np.zeros((num_classes))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            if i.sum() == 0 or num_per_classes[ci] == 0:
                continue

            # 逐步累加, 这里correct里面为true的都是true positive, 即预测的都是真的边框
            fp = (1 - correct[i]).cumsum(0)
            # false的那些预测的都是非真样本, 即false positive
            tp = correct[i].cumsum(0)

            # 当前类别的召回率, tp/当前类别样本数
            r = tp / (num_per_classes[ci] + eps)
            # 拿iou0.5的所有样本召回率
            recall[ci] = r[-1, 0]
            # r[ci] = np.interp(-px, -pred_conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # 当前类别的精度
            p = tp / (tp + fp)
            # 拿iou0.5的所有样本精度
            precision[ci] = p[-1, 0]
            # p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # 每个iou阈值计算AP
            for j in range(tp.shape[1]):
                # 召回率通常是0开始, 精度通常是1开始
                mrec = np.concatenate(([0.0], r[:, j], [1.0]))
                mpre = np.concatenate(([1.0], p[:, j], [0.0]))

                # 对精度从小到大排序再做临近对比
                mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                x = np.linspace(0, 1, 101)
                # 再通过插值得到具体的精度召回率
                ap[ci, j] = np.trapz(np.interp(x, mrec, mpre), x)
        metrics.append([unique_classes, ap, precision, recall])

    return metrics


def box_iou(box1, box2, eps=1E-7):
    """ box 计算iou
    :param box1: [N, 4(x1, y1, x2, y2)]
    :param box2: [M, 4(x1, y1, x2, y2)]
    :return: [N, M]
    """
    box1 = box1[:, None, :]
    xmin = np.maximum(box1[:, :, 0], box2[:, 0])
    ymin = np.maximum(box1[:, :, 1], box2[:, 1])
    xmax = np.minimum(box1[:, :, 2], box2[:, 2])
    ymax = np.minimum(box1[:, :, 3], box2[:, 3])

    w = np.maximum(xmax - xmin, 0)
    h = np.maximum(ymax - ymin, 0)
    inter = w * h
    union = (box1[:, :, 3] - box1[:, :, 1]) * (box1[:, :, 2] - box1[:, :, 0]) + \
            (box2[:, 3] - box2[:, 1]) * (box2[:, 2] - box2[:, 0]) - inter + eps
    return inter / union


def mask_iou(masks1, masks2):
    """ mask计算iou
    :param masks1: [h, w, N]
    :param masks2: [h, w, M]
    :return: [N, M]
    """
    masks_a = np.reshape(np.transpose(masks1, [2, 0, 1]), [masks1.shape[2], -1])
    masks_b = np.reshape(np.transpose(masks2, [2, 0, 1]), [masks2.shape[2], -1])

    intersection = masks_a @ np.transpose(masks_b, [1, 0])
    area_a = np.sum(masks_a, axis=1)
    area_b = np.sum(masks_b, axis=1)

    return intersection / (area_a[:, None] + area_b - intersection)


def val(model, val_data_generator, classes, desc='val'):
    """ 模型评估

    :param model:
    :param val_data_generator:
    :return:
    """
    mAP50, mAP, final_df = 0., 0., []
    stat = []
    iou_vector = np.linspace(0.5, 0.95, 10)
    progress_bar = tqdm.tqdm(range(val_data_generator.total_batch_size), desc=desc, ncols=100)
    for batch in progress_bar:
        # data = val_data_generator.next_batch()
        # valid_nums = data['valid_nums']
        # gt_imgs = np.array(data['imgs'], dtype=np.float32)
        # gt_boxes = np.array(data['bboxes'], dtype=np.float32)
        # gt_classes = data['labels']

        data = val_data_generator.next_batch()
        valid_nums = data['valid_nums']
        gt_imgs = np.array(data['imgs'], dtype=np.float32)
        gt_boxes = data['bboxes']
        gt_labels = data['labels']
        gt_masks = data['masks']

        if model.is_training:
            # yolact_preds = model.model.predict(gt_imgs / 255.)
            yolact_preds = model.model(gt_imgs, training=True)
            out_boxes = decode(yolact_preds[0], model.feature_prior_data)
            print(np.max(yolact_preds[1]))
            out_boxes, out_classes, out_scores, out_masks = detect(
                pred_boxes=out_boxes,
                pred_classes=yolact_preds[1],
                pred_masks=yolact_preds[2],
                pred_proto=yolact_preds[3],
                image_size=model.input_shape[0],
                iou_threshold=0.3,
                conf_thresh=0.4,
                top_k=100
            )
        else:
            predictions = model.model.predict(gt_imgs)
            out_boxes, out_classes, out_scores, out_masks = detect(
                pred_boxes=predictions[0],
                pred_classes=predictions[1],
                pred_masks=predictions[2],
                pred_proto=predictions[3],
                image_size=model.input_shape[0],
                iou_threshold=0.3,
                conf_thresh=0.4,
                top_k=100
            )

        for i in range(len(out_boxes)):
            if out_boxes[i].shape[0]:
                gt_class = gt_labels[i][:valid_nums[i]]
                gt_box = gt_boxes[i][:valid_nums[i], :]
                gt_mask = gt_masks[i][:, :, :valid_nums[i]]

                # [n, m]
                b_iou = box_iou(gt_box, out_boxes[i])
                m_iou = mask_iou(gt_mask, out_masks[i])
                # [n, m]
                correct_label = gt_class[:, None] == out_scores[i]
                # [m, 10]
                b_correct = np.zeros((out_boxes[i].shape[0], iou_vector.shape[0]), dtype=np.bool)
                m_correct = np.zeros((out_boxes[i].shape[0], iou_vector.shape[0]), dtype=np.bool)
                for j, iou_t in enumerate(iou_vector):
                    # 分类正确且iou>阈值
                    xb = np.where((b_iou > iou_t) & correct_label)
                    xm = np.where((m_iou > iou_t) & correct_label)
                    if xb[0].shape[0] and xm[0].shape[0]:
                        b_matches = np.concatenate((np.stack(xb, 1), b_iou[xb[0], xb[1]][:, None]), 1)
                        m_matches = np.concatenate((np.stack(xm, 1), m_iou[xm[0], xm[1]][:, None]), 1)
                        if xb[0].shape[0] > 1 and xm[0].shape[0] > 1:
                            # iou排序
                            b_matches = b_matches[b_matches[:, 2].argsort()[::-1]]
                            m_matches = m_matches[m_matches[:, 2].argsort()[::-1]]
                            # 去重那些 一个预测边框命中多个ground true边框的情况
                            b_matches = b_matches[np.unique(b_matches[:, 1], return_index=True)[1]]
                            m_matches = m_matches[np.unique(m_matches[:, 1], return_index=True)[1]]
                            # 去重那些 一个ground true边框匹配上多个预测边框的情况
                            b_matches = b_matches[np.unique(b_matches[:, 0], return_index=True)[1]]
                            m_matches = m_matches[np.unique(m_matches[:, 0], return_index=True)[1]]
                        b_correct[b_matches[:, 1].astype(int), j] = True
                        m_correct[m_matches[:, 1].astype(int), j] = True

                stat.append((b_correct, m_correct, out_scores[i], out_classes[i], gt_class))
            # tmp_stat = [np.concatenate(x, axis=0) for x in zip(*stat)]
            # ap = ap_per_class(tmp_stat[0], tmp_stat[1], tmp_stat[2], tmp_stat[3])
            # progress_bar.set_postfix(
            #     ordered_dict={"mAP@0.5:0.95": '{:.5f}'.format(ap.mean()), "mAP@0.5": '{:.5f}'.format(ap[:, 0].mean())})

    # 每个类别计算对应的ap
    if stat:
        stat = [np.concatenate(x, axis=0) for x in zip(*stat)]
        box_mask_metrics = ap_per_class(stat[0], stat[1], stat[2], stat[3], stat[4])

        unique_classes, box_ap, box_precision, box_recall = box_mask_metrics[0]
        _, mask_ap, mask_precision, mask_recall = box_mask_metrics[1]

        # AP@0.5, AP@0.5:0.95
        box_mAP50, box_mAP = box_ap[:, 0].mean(), box_ap.mean(1).mean()
        mask_mAP50, mask_mAP = mask_ap[:, 0].mean(), mask_ap.mean(1).mean()

        df = []
        for ci, cls in enumerate(unique_classes):
            if cls != 'None':
                df.append([
                    classes[int(cls)],
                    box_ap[ci, 0], box_ap[ci, :].mean(), box_precision[ci], box_recall[ci],
                    mask_ap[ci, 0], mask_ap[ci, :].mean(), mask_precision[ci], mask_recall[ci]
                ])
        df.append(["total",
                   box_mAP50, box_mAP, box_precision.mean(), box_recall.mean(),
                   mask_mAP50, mask_mAP, mask_precision.mean(), mask_recall.mean()
                   ])
        final_df = pd.DataFrame(data=df, columns=[
            "class",
            'box mAP@0.5', 'box mAP@0.5:0.95', "box precision", "box recall",
            'mask mAP@0.5', 'mask mAP@0.5:0.95', "mask precision", "mask recall",
        ])
        print(final_df)
        mAP50, mAP = box_mAP+mask_mAP, box_mAP50+mask_mAP50
    progress_bar.set_postfix(ordered_dict={"mAP@0.5:0.95": '{:.5f}'.format(mAP), "mAP@0.5": '{:.5f}'.format(mAP50)})
    return mAP50, mAP, final_df


def main():
    # model_path = "h5模型路径, 默认在 ./logs/yolact-best.h5"
    model_path = "./logs/yolact-best.h5"
    # image_path = "提供你要测试的图片路径"
    # image_path = "./data/tmp/Cats_Test49.jpg"
    # image = cv2.imread(image_path)
    val_dataset = './data/instances_train2017.json'
    # image_shape = (640, 640, 3)
    image_shape = (384, 384, 3)
    num_class = 91
    batch_size = 1

    # data generator
    val_coco_data = CoCoDataGenrator(
        coco_annotation_file=val_dataset,
        train_img_nums=500,
        img_shape=image_shape,
        batch_size=batch_size,
        max_instances=num_class,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False
    )

    # 类别名, 也可以自己提供一个数组, 不通过coco
    classes = ['none', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none', 'none', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none',
               'dining table', 'none', 'none', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'none', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    yolact = Yolact(
        input_shape=image_shape,
        num_classes=num_class,
        is_training=False,
        mask_proto_channels=32,
        conf_thres=0.05,
        nms_thres=0.5
    )
    yolact.model.summary(line_length=100)

    mAP50, mAP, metrics = val(model=yolact, val_data_generator=val_coco_data, classes=classes)


if __name__ == "__main__":
    # x = np.array([[0,0,1,1],[1,1,2,2],[2,2,3,3]])
    # y = np.array([[0.5,0.5,1,1],[1.5,1.5,2.5,2.5]])
    # iou = box_iou(x,y)
    # print(iou)
    main()
