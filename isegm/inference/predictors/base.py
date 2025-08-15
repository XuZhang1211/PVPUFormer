import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.engine.trainer import get_next_promts, get_next_promts_inference
from isegm.inference.transforms.zoom_in import get_roi_image_nd
import numpy as np


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 with_sigmoid=True,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=1,
                 **kwargs):
        self.with_flip = with_flip
        self.with_sigmoid = with_sigmoid
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.cascade_step = cascade_step
        self.cascade_adaptive = cascade_adaptive
        self.cascade_clicks = cascade_clicks

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        if self.with_sigmoid:
            self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask=None, on_cascade=False):
        clicks_list = clicker.get_clicks()

        if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
            for i in range(self.cascade_step):
                prediction = self.get_prediction(clicker, None, True)
                if self.cascade_adaptive and prev_mask is not None:
                    diff_num = (
                        (prediction > 0.49) != (prev_mask > 0.49)
                    ).sum()
                    if diff_num <= 20:
                        return prediction
                prev_mask = prediction
            return prediction

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']
    
    def get_vqu_prediction(self, clicker, prev_mask=None, on_cascade=False, gt_mask=None, as_prompt_type=0, click_indx=0, as_multi_prompts=True):
        clicks_list = clicker.get_clicks()

        if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
            for i in range(self.cascade_step):
                prediction, prompts_nd = self.get_vqu_prediction(clicker, None, True, gt_mask, as_prompt_type, click_indx, as_multi_prompts)
                if self.cascade_adaptive and prev_mask is not None:
                    diff_num = (
                        (prediction > 0.49) != (prev_mask > 0.49)
                    ).sum()
                    if diff_num <= 20:
                        return prediction, prompts_nd
                prev_mask = prediction
            return prediction, prompts_nd

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        if as_multi_prompts:
            pred_logits, prompts_nd = self._get_vqu_prediction_prompts(image_nd, clicks_lists, is_image_changed, prev_mask=prev_mask, gt_mask=gt_mask, as_prompt_type=as_prompt_type, click_indx=click_indx)
        else:
            pred_logits, prompts_nd = self._get_vqu_prediction_points(image_nd, clicks_lists, is_image_changed, prev_mask=prev_mask, gt_mask=gt_mask, as_prompt_type=as_prompt_type, click_indx=click_indx)
        
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker), prompts_nd

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0], prompts_nd
    
    def _get_vqu_prediction_points(self, image_nd, clicks_lists, is_image_changed, prev_mask=None, gt_mask=None, as_prompt_type=0, click_indx=0):
        points_nd = self.get_points_nd(clicks_lists)
        gt_mask = np.expand_dims(np.expand_dims(gt_mask, axis=0), axis=0).astype(np.float32)
        if self.with_flip:
            gt_mask = np.concatenate([gt_mask, np.flip(gt_mask, axis=3)], axis=0)
            prev_mask = torch.cat([prev_mask, torch.flip(prev_mask, dims=[3])], dim=0)
        gt_mask = torch.tensor(gt_mask).to(prev_mask.device)
        gt_mask = get_roi_image_nd(gt_mask, self.zoom_in._object_roi, self.zoom_in.target_size)
        prev_mask = get_roi_image_nd(prev_mask, self.zoom_in._object_roi, self.zoom_in.target_size)
        points_nd, prompts_nd = get_next_promts_inference(prev_mask,gt_mask,points_nd, as_allmask=True, jitter_box=True, as_prompt_type=as_prompt_type, click_indx=click_indx)
        return self.net(image_nd, points_nd)['instances'], prompts_nd


    def _get_vqu_prediction_prompts(self, image_nd, clicks_lists, is_image_changed, prev_mask=None, gt_mask=None, as_prompt_type=0, click_indx=0):
        points_nd = self.get_points_nd(clicks_lists)
        gt_mask = np.expand_dims(np.expand_dims(gt_mask, axis=0), axis=0).astype(np.float32)
        if self.with_flip:
            gt_mask = np.concatenate([gt_mask, np.flip(gt_mask, axis=3)], axis=0)
            prev_mask = torch.cat([prev_mask, torch.flip(prev_mask, dims=[3])], dim=0)
            
        gt_mask = torch.tensor(gt_mask).to(prev_mask.device)
        gt_mask = get_roi_image_nd(gt_mask, self.zoom_in._object_roi, self.zoom_in.target_size)
        prev_mask = get_roi_image_nd(prev_mask, self.zoom_in._object_roi, self.zoom_in.target_size)
        prompts_nd = get_next_promts(prev_mask,gt_mask,points_nd, as_allmask=False, jitter_box=False)
        return self.net(image_nd, points_nd, prompts_nd, as_prompt_type)['instances'], prompts_nd
    
    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
