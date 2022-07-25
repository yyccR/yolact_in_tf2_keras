
import random
import json
import numpy as np
from data.generate_coco_data import CoCoDataGenrator
from data.arguments import Arugments

def generate_arguments_data(coco_generator, arguments, argument_percent, output_file):
    """预生成增强数据"""
    from pycocotools import mask

    # 获取最大的image_id
    anno_ids = []
    for k in coco_generator.coco.imgToAnns:
        annos = coco_generator.coco.imgToAnns[k]
        if annos:
            anno_ids.extend(map(lambda x: int(x['id']), annos))
    max_anno_id = max(anno_ids) + 1
    argu_size = int(coco_generator.total_size * argument_percent)

    # 输出json的append到coco_json里再保存
    coco_json_file = coco_generator.coco_annotation_file
    coco_json = json.load(coco_json_file)


    image_id = random.choice(coco_generator.img_ids)
    img = coco_generator._load_image(image_id)
    # {"labels":, "bboxes":, "masks":, "keypoints":}
    annotations = coco_generator._load_annotations(image_id)

    if img.shape[0]:
        r = random.random()
        if r < 0.5:
            input_images = [img]
            input_bboxes = [np.concatenate([annotations['bboxes'], annotations['labels'][:,None]], axis=-1)]
            input_masks = [annotations['masks']]
            # 马赛克4图拼接
            # ids = random.sample(self.img_ids,3)
            while len(input_images) < 4:
                id = random.choice(coco_generator.img_ids)
                img_i = coco_generator._load_image(id)
                if img_i.shape[0]:
                    input_images.append(img_i)
                    annotations_i = coco_generator._load_annotations(id)
                    input_bboxes.append(
                        np.concatenate([annotations_i['bboxes'], annotations_i['labels'][:,None]], axis=-1)
                    )
                    input_masks.append(annotations_i['masks'])

            new_img, new_bboxes, new_masks = \
                arguments.random_mosaic(input_images, input_bboxes, input_masks, coco.img_shape[0])
            if len(new_bboxes)>0:
                pass
                # new_labels = new_bboxes[:,-1]
                # outputs['imgs'] = new_img
                # outputs['labels'] = new_labels
                # outputs['bboxes'] = new_bboxes[:,:-1]
                # for k in range(len(new_bboxes)):

                #
                # if self.include_mask:
                #     outputs['masks'] = new_masks

        else:
            # 平移/旋转/缩放/错切/透视变换
            input_bboxes = np.concatenate([annotations['bboxes'], annotations['labels'][:, None]], axis=-1)
            new_img, new_bboxes, new_masks = arguments.random_perspective(img, input_bboxes, annotations['masks'])
            # 曝光, 饱和, 亮度调整
            new_img = arguments.random_hsv(new_img)
            # if len(new_bboxes)>0:
            #     new_labels = new_bboxes[:, -1]
            #     outputs['imgs'] = new_img
            #     outputs['labels'] = new_labels
            #     outputs['bboxes'] = new_bboxes[:, :-1]
            #     if self.include_mask:
            #         outputs['masks'] = new_masks
    # return outputs


if __name__ == "__main__":
    coco_json = './data/instances_val2017.json'
    # image_shape =
    coco_data = CoCoDataGenrator(
        coco_annotation_file= coco_json,
        train_img_nums=-1,
        img_shape=image_shape,
        batch_size=batch_size,
        max_instances=40,
        include_mask=True,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=True
    )