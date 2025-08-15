from time import time

import numpy as np
import torch
import random
import copy

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    # clicks_list = clicker.get_clicks()
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    as_prompt_type = 0
    as_multi_prompts=True
    iou = 0
    
    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            # as_prompt_type = random.randint(0, 1)
            as_prompt_type = 0
            
            # if click_indx < 5:
            #     as_prompt_type = 0
            # else:
            #     # as_prompt_type = random.randint(0, 1)
            
            #     if np.random.rand() > 0.5:
            #         as_prompt_type = 0
            #     else:
            #         as_prompt_type = 2
                
            # if click_indx == 0:
            #     as_prompt_type = 1
            # else:
            #     as_prompt_type = 0
            
                # as_prompt_type = random.randint(0, 1)
                
            #     if iou > 0.30:
            #         as_prompt_type = 0
            #     else:
            #         as_prompt_type = random.randint(0, 1)
            #         # as_prompt_type = 1
                
            clicker.make_next_click(pred_mask)
            pred_probs, prompts = predictor.get_vqu_prediction(clicker, gt_mask=gt_mask, as_prompt_type=as_prompt_type, click_indx=click_indx, as_multi_prompts=as_multi_prompts)
            pred_mask = pred_probs > pred_thr

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            
            

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, iou,
                            sample_id, click_indx, clicker.clicks_list, True,
                            predictor.zoom_in, prompts, as_prompt_type)
                break

            if callback is not None:
                callback(image, gt_mask, pred_probs, iou,
                         sample_id, click_indx,
                         clicker.clicks_list, False,
                         predictor.zoom_in, prompts, as_prompt_type)

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
