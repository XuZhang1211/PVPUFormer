from time import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
    input_labels = []
    input_points = []
        
    with torch.no_grad():
        # predictor.set_input_image(image)
        # predictor.reset_image()
        predictor.set_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            input_labels.append(int(clicker.clicks_list[click_indx].is_positive))
            input_points.append((clicker.clicks_list[click_indx].coords[1],clicker.clicks_list[click_indx].coords[0]))
            input_label = np.array(input_labels)
            input_point = np.array(input_points)
            
            pred_masks, scores, pred_logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                    return_logits=False,
                )
            pred_mask = pred_masks[0]
            pred_logits = F.Tensor(np.expand_dims(pred_logits, axis=0))
            pred_logits = F.interpolate(pred_logits, pred_mask.shape, mode="bilinear", align_corners=False)
            pred_probs = torch.sigmoid(pred_logits)
            pred_probs = torch.squeeze(pred_probs).numpy()
            
           
            # pred_logits, scores, low_res_masks = predictor.predict(
            #         point_coords=input_point,
            #         point_labels=input_label,
            #         multimask_output=False,
            #         return_logits=True,
            #     )
            # pred_logit = pred_logits[0]
            # pred_logit = F.Tensor(pred_logit)
            # pred_probs = torch.sigmoid(pred_logit)
            # pred_probs = torch.squeeze(pred_probs).numpy()
            # pred_mask = pred_probs > 0.5
            
            # pred_probs = predictor.get_prediction(clicker)
            # pred_mask = pred_probs > 0.5

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, pred_mask, iou,
                            sample_id, click_indx, clicker.clicks_list, True)
                break

            if callback is not None:
                callback(image, gt_mask, pred_probs, pred_mask, iou,
                         sample_id, click_indx,
                         clicker.clicks_list, False)

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
