
from yolact import Yolact
from data.generate_coco_data import CoCoDataGenrator
from data.visual_ops import draw_bounding_box,draw_instance

if __name__ == "__main__":
    image_shape = (384, 384, 3)
    assert (image_shape[0] % 8 == 0) & (image_shape[1] % 8 == 0), "image shape 必须为8的整数倍"
    num_class = 91
    batch_size = 1
    train_img_nums = 1
    train_coco_json = './data/instances_val2017.json'

    yolact = Yolact(
        input_shape=image_shape,
        num_classes=num_class,
        is_training=True,
        mask_proto_channels=32,
        conf_thres=0.05,
        nms_thres=0.5
    )
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
