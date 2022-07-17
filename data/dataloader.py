
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GeneratorDataset(Dataset):
    def __init__(self, coco_generator):
        super(GeneratorDataset, self).__init__()
        self.coco_generator = coco_generator

    def __len__(self):
        return self.coco_generator.total_batch_size

    def __getitem__(self, index):
        batch_img_ids = self.coco_generator.img_ids[index * self.coco_generator.batch_size:
                                                    (index + 1) * self.coco_generator.batch_size]
        batch_imgs = []
        batch_bboxes = []
        batch_labels = []
        batch_masks = []
        batch_keypoints = []
        valid_nums = []

        for img_id in batch_img_ids:
            # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
            data = self.coco_generator._data_generation(image_id=img_id)
            if len(np.shape(data['imgs'])) > 0:
                if len(data['bboxes']) > 0:
                    batch_imgs.append(data['imgs'])
                    batch_labels.append(data['labels'])
                    batch_bboxes.append(data['bboxes'])
                    valid_nums.append(data['valid_nums'])
                    if self.coco_generator.include_mask:
                        if data['masks'].shape == (self.coco_generator.img_shape[0],
                                                   self.coco_generator.img_shape[1],
                                                   self.coco_generator.max_instances):
                            batch_masks.append(data['masks'])

                    if self.coco_generator.include_keypoint:
                        batch_keypoints.append(data['keypoints'])

        while len(batch_imgs) < self.coco_generator.batch_size:

            id = random.choice(self.coco_generator.img_ids)
            img_id = self.img_ids[id]
            data = self.coco_generator._data_generation(image_id=img_id)

            if len(np.shape(data['imgs'])) > 0:
                if len(data['bboxes']) > 0:
                    batch_imgs.append(data['imgs'])
                    batch_labels.append(data['labels'])
                    batch_bboxes.append(data['bboxes'])
                    valid_nums.append(data['valid_nums'])

                    if self.coco_generator.include_mask:
                        if data['masks'].shape == (self.coco_generator.img_shape[0],
                                                   self.coco_generator.img_shape[1],
                                                   self.coco_generator.max_instances):
                            batch_masks.append(data['masks'])

                    if self.coco_generator.include_keypoint:
                        batch_keypoints.append(data['keypoints'])

        output = {
            'imgs': np.array(batch_imgs, dtype=np.int32),
            'bboxes': np.array(batch_bboxes, dtype=np.int16),
            'labels': np.array(batch_labels, dtype=np.int8),
            'masks': np.array(batch_masks, dtype=np.int8),
            'keypoints': np.array(batch_keypoints, dtype=np.int16),
            'valid_nums': np.array(valid_nums, dtype=np.int8)
        }

        return output


def create_dataloader(coco_generator, num_worker):
    dataset = GeneratorDataset(coco_generator)
    return DataLoader(
        dataset=dataset,
        num_workers=num_worker,
        shuffle=True,
        drop_last=True
    )


if __name__ == "__main__":
    from data.generate_coco_data import CoCoDataGenrator
    file = "./instances_val2017.json"
    # file = "./yanhua/annotations.json"
    coco = CoCoDataGenrator(
        coco_annotation_file=file,
        train_img_nums=2,
        max_instances=6,
        include_mask=True,
        include_keypoint=False,
        need_down_image=True,
        batch_size=2,
        using_argument=False
    )
    dataloader = create_dataloader(coco, 2)
    for d in dataloader:
        print(d)