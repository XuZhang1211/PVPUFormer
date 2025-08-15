import cv2
import json
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.pycocotools.coco import COCO
from isegm.data.datasets.tokenizer import tokenize


class CocoDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, word_length=24, cfg=None, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob
        
        if self.cfg is not None:
            dataDir=cfg.COCO_PATH
            dataType='train2017'
            annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
            self.coco_caps=COCO(annFile)

        self.load_samples()

    def load_samples(self):
        annotation_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}'
        self.images_path = self.dataset_path / self.split

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = annotation['annotations']

        self._categories = annotation['categories']
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)
        self._stuff_labels_set = set(self._stuff_labels)

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        if self.copy_paste_prob > 0 or self.image_mix_prob > 0:
            if self.copy_paste_prob > 0:
                # 1. C&P the sample object to another image
                if np.random.rand() < self.copy_paste_prob:
                    # choose an object from sample and put it on target_sample's image
                    target_sample = self.get_random_sample()
                    target_image = target_sample.image  # get image from target_sample

                    # select an object from sample
                    self.points_sampler.sample_object(sample)
                    obj_image = sample.image
                    obj_mask = self.points_sampler.selected_mask[0].astype(int) 
                    obj_indx = obj_mask > 0  # copy object from sample and paste into target_image

                    # apply alpha mixing, put obj on the target image
                    alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                    target_image[obj_indx] = cv2.addWeighted(
                        obj_image, alpha,
                        target_image, 1 - alpha,
                        0
                    )[obj_indx]

                    # construct a new sample
                    sample = DSample(target_image, obj_mask,
                                    objects_ids=[1], sample_id=index)

                # 2. C&P irrelevant object on sample image (do not fully cover the sample object)
                if np.random.rand() < self.copy_paste_prob:
                    # select an object from sample
                    self.points_sampler.sample_object(sample)
                    obj_image = sample.image
                    obj_mask = self.points_sampler.selected_mask[0].astype(int) 
                    obj_indx = obj_mask > 0  # copy object from sample and paste into target_image

                    for _ in range(5):  # at most try n times
                        target_sample = self.get_random_sample()
                        # choose an irrelevant object from the target_sample
                        self.points_sampler.sample_object(target_sample)
                        irr_obj_image = target_sample.image
                        irr_obj_mask = self.points_sampler.selected_mask[0].astype(int)
                        irr_obj_indx = irr_obj_mask > 0

                        if (obj_indx & ~irr_obj_indx).sum() <= 20:  # almost fully convered, retry
                            continue

                        # put mask on it by chance
                        choice = np.random.randint(0, 3)
                        if choice == 0:
                            # put irr_obj in obj image
                            obj_image[irr_obj_indx] = irr_obj_image[irr_obj_indx]
                            obj_mask[irr_obj_indx] = 0  # mask out the irr_obj
                        elif choice == 1:
                            # put irr_obj in obj image
                            obj_image[irr_obj_indx] = irr_obj_image[irr_obj_indx]
                            obj_mask = (obj_mask | irr_obj_mask).astype(int)
                        else:
                            # alpha mixing
                            alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                            obj_image[irr_obj_indx] = cv2.addWeighted(
                                obj_image, alpha,
                                irr_obj_image, 1 - alpha,
                                0
                            )[irr_obj_indx]

                        # construct a new sample
                        sample = DSample(obj_image, obj_mask,
                                        objects_ids=[1], sample_id=index)
                        break

            if np.random.rand() < self.image_mix_prob:
                # randomly select a target image
                target_sample = self.get_random_sample()
                # apply random image mixing augmentation
                # choose an random image and mix with the current target_sample
                alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                sample.image = cv2.addWeighted(
                    sample.image, alpha,
                    target_sample.image, 1 - alpha,
                    0
                )

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())

        # give random order for random points
        # except the first click
        points_pos = points[:, 2] == 100
        indx = np.arange(points_pos.sum()) + 2
        np.random.shuffle(indx)
        points[points_pos, 2] = indx

        mask = self.points_sampler.selected_mask
        
        if self.cfg is not None:
            image_id = int(self.dataset_samples[index][0])
            # load caption annotations
            annIds = self.coco_caps.getAnnIds(imgIds=image_id)
            anns = self.coco_caps.loadAnns(annIds)
            anns_index = random.randrange(0, len(anns))
            caption = anns[anns_index]['caption']
            caption_vec = tokenize(caption, self.word_length, True).squeeze(0)
        else:
            caption_vec = points

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask,
            'captions': caption_vec,
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])
        label_path = self.labels_path / dataset_sample['file_name']

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

        instance_map = np.full_like(label, 0)
        things_ids = []
        stuff_ids = []

        for segment in dataset_sample['segments_info']:
            class_id = segment['category_id']
            obj_id = segment['id']
            if class_id in self._things_labels_set:
                if segment['iscrowd'] == 1:
                    continue
                things_ids.append(obj_id)
            else:
                stuff_ids.append(obj_id)

            instance_map[label == obj_id] = obj_id

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            instances_ids = things_ids + stuff_ids
        else:
            instances_ids = things_ids

            for stuff_id in stuff_ids:
                instance_map[instance_map == stuff_id] = 0

        return DSample(image, instance_map, objects_ids=instances_ids)

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')
