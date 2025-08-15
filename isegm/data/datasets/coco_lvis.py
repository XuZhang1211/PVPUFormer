from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.pycocotools.coco import COCO
from isegm.data.datasets.tokenizer import tokenize


class CocoLvisDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0,
                 allow_list_name=None, anno_file='hannotation.pickle', word_length=24, cfg=None, **kwargs):
        super(CocoLvisDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob
        self.word_length = word_length
        self.cfg = cfg

        with open(self._split_path / anno_file, 'rb') as f:
            self.dataset_samples = sorted(pickle.load(f).items())
            
        if self.cfg is not None:
            dataDir=cfg.COCO_PATH
            dataType='train2017'
            annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
            self.coco_caps=COCO(annFile)

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                    if sample[0] in allow_images_ids]

    
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
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f'{image_id}.jpg'

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f'{image_id}.pickle'
        with open(packed_masks_path, 'rb') as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample['hierarchy'])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                instances_info[inst_id] = {
                    'mapping': objs_mapping[inst_id],
                    'parent': None,
                    'children': []
                }
        else:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        return DSample(image, layers, objects=instances_info)
