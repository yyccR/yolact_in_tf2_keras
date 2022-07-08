import sys

sys.path.append('../yolact_in_tf2_keras')

import os
import tqdm
import numpy as np
import random
import tensorflow as tf
from data.visual_ops import draw_bounding_box, draw_instance
from data.generate_coco_data import CoCoDataGenrator
from loss import MultiBoxLoss
from yolact import Yolact
from box_utils import detect,decode

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    epochs = 300
    log_dir = './logs'
    image_shape = (384, 384, 3)
    assert (image_shape[0] % 8 == 0) & (image_shape[1] % 8 == 0), "image shape 必须为8的整数倍"
    # 类别数
    num_class = 91
    batch_size = 5
    # -1表示全部数据参与训练
    train_img_nums = 5
    train_coco_json = './data/instances_val2017.json'
    val_coco_json = './data/instances_val2017.json'

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

    # 这里anchor归一化到[0,1]区间
    # anchors = np.array([[10, 13], [16, 30], [33, 23],
    #                     [30, 61], [62, 45], [59, 119],
    #                     [116, 90], [156, 198], [373, 326]]) / 640.
    # anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    # anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # tensorboard日志
    summary_writer = tf.summary.create_file_writer(log_dir)
    # data generator
    coco_data = CoCoDataGenrator(
        coco_annotation_file= train_coco_json,
        train_img_nums=train_img_nums,
        img_shape=image_shape,
        batch_size=batch_size,
        include_mask=True,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=False
    )
    # 验证集
    # val_coco_data = CoCoDataGenrator(
    #     coco_annotation_file=val_coco_json,
    #     train_img_nums=-1,
    #     img_shape=image_shape,
    #     batch_size=batch_size,
    #     include_mask=False,
    #     include_crowd=False,
    #     include_keypoint=False,
    #     need_down_image=False,
    #     using_argument=False
    # )

    yolact = Yolact(
        input_shape=image_shape,
        num_classes=num_class,
        is_training=True,
        mask_proto_channels=32,
        conf_thres=0.05,
        nms_thres=0.5
    )
    yolact.model.summary(line_length=200)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    loss_fn = MultiBoxLoss(
        batch_size=batch_size,
        num_classes=num_class,
        priors=yolact.feature_prior_data,
        conf_alpha=3,
        bbox_alpha=1,
        semantic_segmentation_alpha=1,
        mask_alpha=6.125,
        pos_thresh=0.5,
        neg_thresh=0.4,
        masks_to_train=100,
        ohem_negpos_ratio=2
    )

    pre_mAP = 0.
    # data = coco_data.next_batch()
    for epoch in range(epochs):
        train_progress_bar = tqdm.tqdm(range(coco_data.total_batch_size), desc="train epoch {}/{}".format(epoch, epochs-1), ncols=100)
        for batch in train_progress_bar:
            with tf.GradientTape() as tape:
                data = coco_data.next_batch()
                valid_nums = data['valid_nums']
                gt_imgs = np.array(data['imgs'] / 255., dtype=np.float32)
                gt_boxes = [data['bboxes'][i][:valid_num] / image_shape[0] for i,valid_num in enumerate(valid_nums)]
                gt_labels = [data['labels'][i][:valid_num] for i,valid_num in enumerate(valid_nums)]
                gt_masks = [data['masks'][i][:,:,:valid_num] for i,valid_num in enumerate(valid_nums)]

                # print("-------epoch {}, step {}, total step {}--------".format(epoch, batch,
                #                                                                epoch * coco_data.total_batch_size + batch))
                # print("current data index: ",
                #       coco_data.img_ids[(coco_data.current_batch_index - 1) * coco_data.batch_size:
                #                         coco_data.current_batch_index * coco_data.batch_size])
                # for i, nums in enumerate(valid_nums):
                #     print("gt boxes: ", gt_boxes[i, :nums, :] * image_shape[0])
                #     print("gt classes: ", gt_classes[i, :nums])

                yolact_preds = yolact.model(gt_imgs, training=True)
                box_loss, mask_loss, cls_loss, total_loss = loss_fn(yolact_preds, gt_boxes, gt_labels, gt_masks)

                train_progress_bar.set_postfix(ordered_dict={"loss":'{:.5f}'.format(total_loss)})

                grad = tape.gradient(total_loss, yolact.model.trainable_variables)
                optimizer.apply_gradients(zip(grad, yolact.model.trainable_variables))

                # Scalar
                with summary_writer.as_default():
                    tf.summary.scalar('loss/box_loss', box_loss,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/mask_loss', mask_loss,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/class_loss', cls_loss,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/total_loss', total_loss,
                                      step=epoch * coco_data.total_batch_size + batch)

                # image, 只拿每个batch的其中一张
                random_one = random.choice(range(batch_size))
                # gt
                gt_img = gt_imgs[random_one].copy() * 255
                gt_box = gt_boxes[random_one] * image_shape[0]
                gt_class = gt_labels[random_one]
                gt_mask = gt_masks[random_one]
                non_zero_ids = np.where(np.sum(gt_box, axis=-1))[0]
                for i in non_zero_ids:
                    cls = gt_class[i]
                    class_name = coco_data.coco.cats[cls]['name']
                    xmin, ymin, xmax, ymax = gt_box[i]
                    # print(xmin, ymin, xmax, ymax)
                    gt_img = draw_bounding_box(gt_img, class_name, cls, int(xmin), int(ymin), int(xmax), int(ymax))
                    gt_img = draw_instance(gt_img, gt_mask, alpha=0.3)

                # pred, 同样只拿随机一个batch的pred
                pred_img = gt_imgs[random_one].copy() * 255
                out_boxes = decode(yolact_preds[0], yolact.feature_prior_data)
                out_boxes, out_classes, out_scores, out_masks = detect(
                    pred_boxes=out_boxes,
                    pred_classes=yolact_preds[1],
                    pred_masks=yolact_preds[2],
                    pred_proto=yolact_preds[3],
                    image_size=image_shape[0],
                    iou_threshold=0.3,
                    conf_thresh=0.4,
                    top_k=100
                )
                if len(out_classes) == batch_size:
                    random_boxes = out_boxes[random_one]
                    random_cls_scores = out_scores[random_one]
                    random_labels = out_classes[random_one]
                    for i, box_obj_cls in enumerate(random_boxes):
                        if random_cls_scores[i] > 0.6:
                            label = int(random_labels[i])
                            if coco_data.coco.cats.get(label):
                                class_name = coco_data.coco.cats[label]['name']
                                # class_name = classes[label]
                                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                                pred_img = draw_bounding_box(pred_img, class_name, random_cls_scores[i], int(xmin), int(ymin),
                                                             int(xmax), int(ymax))
                                pred_img = draw_instance(pred_img, out_masks[random_one][:,:,i], alpha=0.3)

                concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                summ_imgs = tf.expand_dims(concat_imgs, 0)
                summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                with summary_writer.as_default():
                    tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                     step=epoch * coco_data.total_batch_size + batch)
        # 这里计算一下训练集的mAP
        # val(model=yolo, val_data_generator=coco_data, classes=classes, desc='training dataset val')
        # # 这里计算验证集的mAP
        # mAP50, mAP, final_df = val(model=yolo, val_data_generator=val_coco_data, classes=classes, desc='val dataset val')
        # if mAP > pre_mAP:
        #     pre_mAP = mAP
        #     yolo.yolov5.save_weights(log_dir+"/yolov{}-best.h5".format(yolov5_type))
        #     print("save {}/yolov{}-best.h5 best weight with {} mAP.".format(log_dir, yolov5_type, mAP))
        # yolo.yolov5.save_weights(log_dir+"/yolov{}-last.h5".format(yolov5_type))
        # print("save {}/yolov{}-last.h5 last weights at epoch {}.".format(log_dir, yolov5_type, epoch))


if __name__ == "__main__":
    main()
