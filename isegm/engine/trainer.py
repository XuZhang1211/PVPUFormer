import os
import random
import logging
import pprint
from copy import deepcopy
from collections import defaultdict
import bezier
from scipy.interpolate import make_interp_spline

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 use_iterloss=False,
                 iterloss_weights=None,
                 use_random_clicks=True,
                 iter_train='epochiter',
                 penalty_loss=False,
                 ed_loss=False,
                 pclout=False,
                 as_multi_prompts_ed_loss=False,
                 as_allmask=True,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks
        self.current_epoch = 0
        self.iter_train = iter_train

        # iterloss
        self.use_iterloss = use_iterloss
        self.iterloss_weights = iterloss_weights
        self.use_random_clicks = use_random_clicks
        self.penalty_loss = penalty_loss
        self.ed_loss = ed_loss
        self.pclout = pclout
        self.as_multi_prompts_ed_loss = as_multi_prompts_ed_loss
        self.as_allmask = as_allmask

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))
            logger.info('Run experiment with config:')
            logger.info(pprint.pformat(cfg, indent=4))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.scaler: torch.cuda.amp.GradScaler

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')

        if self.cfg.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            accumulate_grad = ((i + 1) % self.cfg.accumulate_grad == 0) or \
                (i + 1 == len(self.train_data))

            if self.cfg.amp:
                loss /= self.cfg.accumulate_grad
                self.scaler.scale(loss).backward()
                if accumulate_grad:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
            else:
                loss.backward()
                if accumulate_grad:
                    self.optim.step()
                    self.optim.zero_grad()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                # tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                
                if not hasattr(self, 'lr_scheduler'):
                    lr = self.lr
                else:
                    lr = self.lr_scheduler.get_lr()[-1]
                
                iou_value = format(metric.get_epoch_value(),'0.5f')
                tbar.set_description(f'Epoch {epoch} lr {lr}, training loss {train_loss/(i+1):.4f}, {log_prefix}-Metrics/{metric.name}: {iou_value}')
                
                
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)
                
                if not hasattr(self, 'lr_scheduler'):
                    lr = self.lr
                else:
                    lr = self.lr_scheduler.get_lr()[-1]
                
                iou_value = format(metric.get_epoch_value(),'0.5f')
                tbar.set_description(f'Epoch {epoch} lr {lr}, training loss {train_loss/(i+1):.4f}, {log_prefix}-Metrics/{metric.name}: {iou_value}')
                
            # save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
            #                 epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()
        if self.iter_train == 'epochiter':
            as_text = self.current_epoch % 2 != 0
        else:
            as_text = np.random.rand() > 0.5

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            if batch_data['captions'] is not None:
                captions = batch_data['captions']

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            loss = 0.0
            
            
            # if self.ed_loss:
            ed_pos_label = batch_data['instances'].repeat(1, self.max_interactive_points, 1, 1) # B, N, H, W
            ed_neg_label = torch.logical_not(batch_data['instances']).repeat(1, self.max_interactive_points, 1, 1) # B, N, H, W
            ed_mask_label = torch.concatenate([ed_pos_label, ed_neg_label], dim=1)

            if not self.use_random_clicks:
                points[:] = -1
                points = get_next_points(prev_output,
                                         gt_mask,
                                         points)

            num_iters = random.randint(1, self.max_num_next_clicks)

            if self.use_iterloss:
                # iterloss
                for click_indx in range(num_iters):
                    if click_indx == 0 and as_text:
                        as_click = False
                    else:
                        as_click = True
                    
                    # if click_indx < 1:
                    #     as_prompt_type = 0
                    #     # if np.random.rand() > 0.5:
                    #     #     as_prompt_type = 0
                    #     # else:
                    #     #     as_prompt_type = random.randint(0, 2)
                    #     # as_prompt_type = random.randint(0, 2)
                    # else:
                    #     as_prompt_type = random.randint(0, 1)
                        
                    # if np.random.rand() > 0.5:
                    #     as_prompt_type = 0
                    # else:
                    #     as_prompt_type = 2
                    
                    # as_prompt_type = random.randint(0, 2)
                    
                    # as_prompt_type = 0
                    as_prompt_type = random.randint(0, 1)
                    
                    if click_indx == 0:
                        if self.as_multi_prompts_ed_loss:
                            # multi-type prompt 
                            _, next_boxes, scribbles, _ = get_next_promts(prev_output,
                                                        gt_mask,
                                                        points,
                                                        ed_mask_label,
                                                        as_allmask=self.as_allmask)
                            
                            prompts = [points, next_boxes, scribbles]
                
                    # v1
                    # net_input = torch.cat((image, prev_output), dim=1) \
                    #     if self.net.with_prev_mask else image
                    # v2
                    net_input = torch.cat((image, prev_output.detach()), dim=1) \
                        if self.net.with_prev_mask else image
                    
                    # if len(batch_data['captions'].shape) != len(points.shape):
                    #     # click, text co-training
                    #     output = self._forward(self.net, net_input, points, captions, as_click)
                    
                    if self.as_multi_prompts_ed_loss:
                        output = self._forward(self.net, net_input, points, prompts, as_prompt_type, self.ed_loss, self.pclout)
                    elif len(batch_data['captions'].shape) != len(points.shape):
                        # click, text co-training
                        output = self._forward(self.net, net_input, points, captions, as_click)
                    else:
                        output = self._forward(self.net, net_input, points)
                    
                    loss = self.add_loss(
                        'instance_loss', loss, losses_logging, validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    loss = self.add_loss(
                        'instance_aux_loss', loss, losses_logging, validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    loss = self.add_loss(
                        'instance_aux2_loss', loss, losses_logging, validation,
                        lambda: (output['instances_aux'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    if self.ed_loss or self.as_multi_prompts_ed_loss:
                        loss = self.add_loss(
                            'instance_aux3_loss', loss, losses_logging, validation,
                            lambda: (output['instances_aux'], ed_mask_label),
                            iterloss_step=click_indx,
                            iterloss_weight=self.iterloss_weights[click_indx])

                    # prev_output = torch.sigmoid(output['instances'])
                    # if click_indx < num_iters - 1:
                    #     points = get_next_points(prev_output,
                    #                             gt_mask,
                    #                             points)
                    
                    if not self.pclout:
                        prev_output = torch.sigmoid(output['instances'])
                    else:
                        prev_output = output['instances']
                        
                    if click_indx < num_iters - 1:
                        
                        
                        if self.ed_loss and self.as_multi_prompts_ed_loss:
                            # multi-type prompt 
                            points, next_boxes, scribbles, ed_mask_label = get_next_promts(prev_output,
                                                        gt_mask,
                                                        points,
                                                        ed_mask_label,
                                                        as_allmask=self.as_allmask)
                            prompts = [points, next_boxes, scribbles]
                        elif self.ed_loss and not self.as_multi_prompts_ed_loss:
                            points, ed_mask_label = get_next_points_and_mask(prev_output,
                                                    gt_mask,
                                                    points, ed_mask_label)
                        else:
                            points = get_next_points(prev_output,
                                                    gt_mask,
                                                    points)

                    if self.net.with_prev_mask and self.prev_mask_drop_prob > 0:
                        zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                        prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            else:
                # iter mask (RITM)
                points, prev_output = self.find_next_n_points(
                    image,
                    gt_mask,
                    points,
                    prev_output,
                    num_iters,
                    not validation
                )

                net_input = torch.cat((image, prev_output), dim=1) \
                    if self.net.with_prev_mask else image
                output = self._forward(self.net, net_input, points)

                loss = self.add_loss(
                    'instance_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances'], batch_data['instances']))
                loss = self.add_loss(
                    'instance_aux_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances'], batch_data['instances']))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))

        batch_data['points'] = points
        return loss, losses_logging, batch_data, output

    def find_next_n_points(self, image, gt_mask, points, prev_output,
                           num_points, eval_mode=False, grad=False):
        with torch.set_grad_enabled(grad):
            for _ in range(num_points):

                if eval_mode:
                    self.net.eval()

                net_input = torch.cat((image, prev_output), dim=1) \
                    if self.net.with_prev_mask else image
                prev_output = torch.sigmoid(
                    self._forward(
                        self.net,
                        net_input,
                        points
                    )['instances']
                )

                points = get_next_points(prev_output, gt_mask, points)

                if eval_mode:
                    self.net.train()

            if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and num_points > 0:
                zero_mask = np.random.random(
                    size=prev_output.size(0)) < self.prev_mask_drop_prob
                prev_output[zero_mask] = \
                    torch.zeros_like(prev_output[zero_mask])
        return points, prev_output

    def _forward(self, model, net_input, points, captions, as_prompt_type, *args, **kwargs):
        # handle autocast for automatic mixed precision
        if self.cfg.amp:
            with torch.cuda.amp.autocast():
                output = model(net_input, points, captions, as_prompt_type, *args, **kwargs)
        else:
            
            output = model(net_input, points, captions, as_prompt_type, *args, **kwargs)
        return output

    def add_loss(self, loss_name, total_loss, losses_logging, validation,
                 lambda_loss_inputs, iterloss_step=None, iterloss_weight=1):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)

            if iterloss_step is not None:
                losses_logging[
                    loss_name + f'_{iterloss_step}_{iterloss_weight}'
                ] = loss 
                loss = loss_weight * loss * iterloss_weight
            else:
                # iter mask (RITM)
                losses_logging[loss_name] = loss
                loss = loss_weight * loss

            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (255, 0, 0))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, pred_thresh=0.49):
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)

    return points

def get_next_points_and_mask(pred, gt, points, ed_mask_label, pred_thresh=0.49):
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)
    
    error_mask = np.logical_or(fn_mask, fp_mask)
    fn_mask_ = torch.from_numpy(fn_mask.astype(np.int32)).type_as(ed_mask_label).to(ed_mask_label.device)
    fp_mask_ = torch.from_numpy(fp_mask.astype(np.int32)).type_as(ed_mask_label).to(ed_mask_label.device)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
                ed_mask_label[bindx, loc] = fn_mask_[bindx]
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
                ed_mask_label[bindx, loc] = fp_mask_[bindx]
    return points, ed_mask_label


def get_next_promts(pred, gt, points, ed_mask_label=None, pred_thresh=0.49, as_allmask=False, jitter_box=True):
    if isinstance(gt, torch.Tensor):
        gt_mask_np = gt.cpu().numpy()[:, 0, :, :]
        gt = gt.cpu().numpy()[:, 0, :, :] > 0.5
    else:
        if len(gt) != len(pred):
            gt = np.expand_dims(gt,axis=0)
            gt = gt > 0.5
        else:
            gt = gt[:, 0, :, :] > 0.5
        
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    
    # fix bug: -> pred_thresh被赋值为(click_indx+1)
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)
    error_mask = np.logical_or(fn_mask, fp_mask)
    if ed_mask_label is not None:
        fn_mask_ = torch.from_numpy(fn_mask.astype(np.int32)).type_as(ed_mask_label).to(ed_mask_label.device)
        fp_mask_ = torch.from_numpy(fp_mask.astype(np.int32)).type_as(ed_mask_label).to(ed_mask_label.device)
    
    next_boxes = cal_box(gt, fn_mask, fp_mask, points, as_allmask=as_allmask, jitter_box=jitter_box)
    scribbles = cal_scribble(gt, min_p=3, max_p=10, num_samples = 1000)
    
    next_boxes = torch.from_numpy(next_boxes).to(points.device)
    # scribbles = torch.from_numpy(scribbles).to(points.device)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
                if ed_mask_label is not None:
                    ed_mask_label[bindx, loc] = fn_mask_[bindx]
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
                if ed_mask_label is not None:
                    ed_mask_label[bindx, loc] = fp_mask_[bindx]
    if ed_mask_label is not None:
        return points, next_boxes, scribbles, ed_mask_label
    else:
        return points, next_boxes, scribbles

def cal_box_inference(gt_mask, fn_mask, fp_mask, points, as_allmask=True, jitter_box=True, set_offset=10):
    mask_height, mask_width = gt_mask.shape[1], gt_mask.shape[2]
    num_points = points.size(1) // 2
    # error_mask = np.logical_or(fn_mask, fp_mask)
    next_boxes = np.zeros([len(fn_mask), 5], np.int32)
    next_boxes_points = np.zeros([len(fn_mask), 6], np.int32)
    for bindx in range(len(fn_mask)):
        if as_allmask:
            mask_ind = np.argwhere(gt_mask[bindx]==True)
            is_positive = True
            loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
            loc = loc[0, 0] if len(loc) > 0 else num_points - 1
        else:
            is_positive = np.sum(fn_mask[bindx]) > np.sum(fp_mask[bindx])
            if is_positive:
                next_mask = max_connected_regions(fn_mask[bindx])
                # next_mask = fn_mask[bindx]
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                # loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                loc = num_points - 1
            else:
                next_mask = max_connected_regions(fp_mask[bindx])
                # next_mask = fp_mask[bindx]
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
            
            mask_ind = np.argwhere(next_mask==True)
        if len(mask_ind) > 0:
            y0, y1, x0, x1 = mask_ind[:,0].min(), mask_ind[:,0].max(), mask_ind[:,1].min(), mask_ind[:,1].max()
            
            intra_point_ind = random.randint(0, len(mask_ind)-1)
            intra_point_y, intra_point_x = mask_ind[intra_point_ind,0], mask_ind[intra_point_ind,1]
            
            # jitter_ratio: 0.02
            if jitter_box:
                offset = random.randint(-set_offset, 0)
                begin_x = np.minimum(np.maximum(x0 + offset, 0), mask_width-set_offset)
                offset = random.randint(0, set_offset)
                end_x = np.maximum(np.minimum(x1 + offset, mask_width), begin_x + set_offset)
                
                offset = random.randint(-set_offset, 0)
                begin_y = np.minimum(np.maximum(y0 + offset, 0), mask_height-set_offset)
                offset = random.randint(0, set_offset)
                end_y = np.maximum(np.minimum(y1 + offset, mask_height), begin_y + set_offset)
                
                y0, y1, x0, x1 = begin_y, end_y, begin_x, end_x
            
            x_center = int(0.5*(x0+x1))
            y_center = int(0.5*(y0+y1))
            b_width = int(x1 - x0)
            b_height = int(y1 - y0)
            
            if x_center not in mask_ind and y_center not in mask_ind:
                x_center = intra_point_x
                y_center = intra_point_y
            try:
                assert x_center >= 1
                assert y_center >= 1
                assert b_width >= 1
                assert b_height >= 1

                # x_center, y_center, b_width, b_height, prompts_index
                label_ind = loc.cpu().numpy() if isinstance(loc, torch.Tensor) else loc
                next_boxes[bindx] = [x_center, y_center, b_width, b_height, label_ind]
                next_boxes_points[bindx] = [y0, x0, y1, x1, y_center, x_center]
            except:
                next_boxes[bindx] = [0, 0, 0, 0, 0]
                next_boxes_points[bindx] = [0, 0, 0, 0, 0, 0]
        else:
            next_boxes[bindx] = [0, 0, 0, 0, 0]
            next_boxes_points[bindx] = [0, 0, 0, 0, 0, 0]
        
    return next_boxes, next_boxes_points

def cal_scribble_inference(gt_mask, min_p=3, max_p=10, num_samples = 1000):
    bs_scribbles = []
    bs_bounding_rectangles = []
    bs_scribbles_points = []
    for i in range(len(gt_mask)):
        if np.sum(gt_mask[i]) > 0:
            mask = max_connected_regions(gt_mask[i])
            mask_ind_hw = np.argwhere(mask==True)
            # num_p = random.randint(min_p, max_p)
            num_p = max_p
            # if len(mask_ind_hw) > 0:
            x0, x1, y0, y1 = mask_ind_hw[:,0].min(), mask_ind_hw[:,0].max(), mask_ind_hw[:,1].min(), mask_ind_hw[:,1].max()
            
            x_center = int(0.5*(x0+x1))
            y_center = int(0.5*(y0+y1))
            b_width = int(x1 - x0)
            b_height = int(y1 - y0)
            # bounding_rectangle = np.array([[x_center, y_center, b_width, b_height]])
            bounding_rectangle = np.array([[y_center, x_center, b_height, b_width]])
            bbox=[x0, y0, x1, y1]
            
            value = x0
            gap = b_width // num_p
            s_points = []
            for ind in range(num_p):
                if value+gap > value:
                    x_point = random.randint(value, value+gap-1)
                else:
                    x_point = random.randint(value, value+gap)
                x_ind = mask_ind_hw[:,0] == x_point
                if mask_ind_hw[x_ind].shape[0] > 0:
                    y_ind = random.randint(0, mask_ind_hw[x_ind].shape[0]-1)
                    s_points.append(mask_ind_hw[x_ind][y_ind])
                value += gap
                
            points = np.array(s_points)
            scribbles_points = points.copy()
            if len(points) > 0:
                as_inline = np.random.rand() > 0.5
                # as_inline = False
                # scribbles = bezier_curve(points, num_samples, as_inline=as_inline)
                scribbles = bezier_curve(points, bbox, num_samples, as_inline=as_inline)
                scribbles = scribbles[:,::-1]
            else:
                scribbles = np.zeros([num_samples, 2])
                bounding_rectangle = np.array([[0, 0, 0, 0]])
        else:
            scribbles = np.zeros([num_samples, 2])
            bounding_rectangle = np.array([[0, 0, 0, 0]])
        bs_scribbles.append(np.expand_dims(scribbles, 0))
        bs_bounding_rectangles.append(bounding_rectangle)
        bs_scribbles_points.append(np.expand_dims(scribbles_points, 0))
        
    bs_scribbles = np.expand_dims(np.concatenate(bs_scribbles, 0), 1)
    bs_bounding_rectangles = np.array(bs_bounding_rectangles)
    bs_scribbles_points = np.expand_dims(np.concatenate(bs_scribbles_points, 0), 1)
    return [bs_scribbles, bs_bounding_rectangles], bs_scribbles_points

def get_next_promts_inference(pred, gt, points, ed_mask_label=None, pred_thresh=0.49, as_allmask=True, jitter_box=True, as_prompt_type=0, click_indx=0):
    if isinstance(gt, torch.Tensor):
        gt_mask_np = gt.cpu().numpy()[:, 0, :, :]
        gt = gt.cpu().numpy()[:, 0, :, :] > 0.5
    else:
        if len(gt) != len(pred):
            gt = np.expand_dims(gt,axis=0)
            gt = gt > 0.5
        else:
            gt = gt[:, 0, :, :] > 0.5
        
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    
    # fix bug: -> pred_thresh被赋值为(click_indx+1)
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)
    error_mask = np.logical_or(fn_mask, fp_mask)
    
    next_boxes, next_boxes_points = cal_box_inference(gt, fn_mask, fp_mask, points, as_allmask=as_allmask, jitter_box=jitter_box)
    scribbles, scribbles_points = cal_scribble_inference(gt, min_p=3, max_p=7, num_samples = 1000)
    next_boxes = torch.from_numpy(next_boxes).to(points.device)
    next_boxes_points = torch.from_numpy(next_boxes_points).to(points.device)
    
    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points_copy = points.clone()
    
        
                    
    if as_prompt_type == 0:
        
        # for bindx in range(fn_mask.shape[0]):
        #     fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        #     fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        #     fn_max_dist = np.max(fn_mask_dt)
        #     fp_max_dist = np.max(fp_mask_dt)

        #     is_positive = fn_max_dist > fp_max_dist
        #     dt = fn_mask_dt if is_positive else fp_mask_dt
        #     inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        #     indices = np.argwhere(inner_mask)
        #     if len(indices) > 0:
        #         coords = indices[np.random.randint(0, len(indices))]
        #         order = max(points_copy[bindx, :, 2].max(), 0) + 1
        #         if is_positive:
        #             loc = torch.argwhere(points_copy[bindx, :num_points, 2] < 0)
        #             loc = loc[0, 0] if len(loc) > 0 else num_points - 1
        #             points_copy[bindx, loc, 0] = float(coords[0])
        #             points_copy[bindx, loc, 1] = float(coords[1])
        #             points_copy[bindx, loc, 2] = float(order)
        #         else:
        #             loc = torch.argwhere(points_copy[bindx, num_points:, 2] < 0)
        #             loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
        #             points_copy[bindx, loc, 0] = float(coords[0])
        #             points_copy[bindx, loc, 1] = float(coords[1])
        #             points_copy[bindx, loc, 2] = float(order)
                    
        points_vpu = points_copy
        
    if as_prompt_type == 1:
        
        if torch.sum(next_boxes_points) == 0:
            points_vpu = points_copy
        else:
            if click_indx == 0:
                points_box = torch.zeros([points_copy.shape[0],4,3]).to(points_copy.device) - 1
                for bindx in range(points_copy.shape[0]):
                    if torch.sum(next_boxes_points[bindx]) == 0:
                        continue
                    points_box[bindx, 0, 0] = float(next_boxes_points[bindx][4])
                    points_box[bindx, 0, 1] = float(next_boxes_points[bindx][5])
                    points_box[bindx, 0, 2] = float(1)
                    
                    points_box[bindx, 2, 0] = float(next_boxes_points[bindx][0])
                    points_box[bindx, 2, 1] = float(next_boxes_points[bindx][1])
                    points_box[bindx, 2, 2] = float(0)
                    points_box[bindx, 3, 0] = float(next_boxes_points[bindx][2])
                    points_box[bindx, 3, 1] = float(next_boxes_points[bindx][3])
                    points_box[bindx, 3, 2] = float(2)
                points_vpu = points_box
            else:
                points_box = torch.zeros([points_copy.shape[0],4,3]).to(points_copy.device) - 1
                for bindx in range(points_copy.shape[0]):
                    if torch.sum(next_boxes_points[bindx]) == 0:
                        continue
                    order = max(points_copy[bindx, :, 2].max(), 0) + 1
                    
                    points_box[bindx, 0, 0] = float(next_boxes_points[bindx][4])
                    points_box[bindx, 0, 1] = float(next_boxes_points[bindx][5])
                    points_box[bindx, 0, 2] = float(order+1)
                    
                    points_box[bindx, 2, 0] = float(next_boxes_points[bindx][0])
                    points_box[bindx, 2, 1] = float(next_boxes_points[bindx][1])
                    points_box[bindx, 2, 2] = float(order+0)
                    points_box[bindx, 3, 0] = float(next_boxes_points[bindx][2])
                    points_box[bindx, 3, 1] = float(next_boxes_points[bindx][3])
                    points_box[bindx, 3, 2] = float(order+2)
                    
                pos_clicks1 = points_copy[:,:num_points]
                pos_clicks2 = points_box[:,:2]
                pos_click = torch.concatenate([pos_clicks1, pos_clicks2], dim=1)
                
                neg_clicks1 = points_copy[:,num_points:]
                neg_clicks2 = points_box[:,2:]
                neg_clicks = torch.concatenate([neg_clicks1, neg_clicks2], dim=1)
                points_vpu = torch.concatenate([pos_click, neg_clicks], dim=1)
    
    if as_prompt_type == 2:
        
        if click_indx == 0:
            points_scribbles = torch.zeros([points_copy.shape[0],scribbles_points.shape[2]*2,3]).to(points_copy.device) - 1
            for bindx in range(points_copy.shape[0]):
                for p_indx in range(scribbles_points.shape[2]):
                    points_scribbles[bindx, p_indx, 0] = float(scribbles_points[bindx][0][p_indx][0])
                    points_scribbles[bindx, p_indx, 1] = float(scribbles_points[bindx][0][p_indx][1])
                    points_scribbles[bindx, p_indx, 2] = float(p_indx)
                
            points_vpu = points_scribbles
            
        else:
            points_scribbles = torch.zeros([points_copy.shape[0],scribbles_points.shape[2]*2,3]).to(points_copy.device) - 1
            for bindx in range(points_copy.shape[0]):
                order = max(points_copy[bindx, :, 2].max(), 0) + 1
                for p_indx in range(scribbles_points.shape[2]):
                    points_scribbles[bindx, p_indx, 0] = float(scribbles_points[bindx][0][p_indx][0])
                    points_scribbles[bindx, p_indx, 1] = float(scribbles_points[bindx][0][p_indx][1])
                    points_scribbles[bindx, p_indx, 2] = float(order+p_indx)
                    
            
            pos_clicks1 = points_copy[:,:num_points]
            pos_clicks2 = points_scribbles[:,:scribbles_points.shape[2]]
            pos_click = torch.concatenate([pos_clicks1, pos_clicks2], dim=1)
            
            neg_clicks1 = points_copy[:,num_points:]
            neg_clicks2 = points_scribbles[:,scribbles_points.shape[2]:]
            neg_clicks = torch.concatenate([neg_clicks1, neg_clicks2], dim=1)
            
            points_vpu = torch.concatenate([pos_click, neg_clicks], dim=1)
            
    return points_vpu, [points_vpu, next_boxes, scribbles]

def get_iou(pred, gt, pred_thresh=0.49):
    pred_mask = pred > pred_thresh
    gt_mask = gt > 0.5

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)


def cal_box(gt_mask, fn_mask, fp_mask, points, as_allmask=True, jitter_box=True, set_offset=10):
    # def draw_boxs(image, boxs):
    #     image = image.copy()
    #     box = boxs[0]
    #     x_center, y_center, b_width, b_height, boxes_index = box.astype('int32')
    #     x0,x1,y0,y1 = x_center-b_width//2, x_center+b_width//2, y_center-b_height//2, y_center+b_height//2
    #     image = np.uint8((image.astype(int))*255)
    #     cv2.rectangle(image, (x0, y0), (x1,y1),  (255, 255, 255), 3)
    #     cv2.imwrite('./results/plt_box_mask.jpg',image)
        # return image
    mask_height, mask_width = gt_mask.shape[1], gt_mask.shape[2]
    num_points = points.size(1) // 2
    # error_mask = np.logical_or(fn_mask, fp_mask)
    next_boxes = np.zeros([len(fn_mask), 5], np.int32)
    for bindx in range(len(fn_mask)):
        if as_allmask:
            mask_ind = np.argwhere(gt_mask[bindx]==True)
            is_positive = True
            loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
            loc = loc[0, 0] if len(loc) > 0 else num_points - 1
        else:
            is_positive = np.sum(fn_mask[bindx]) > np.sum(fp_mask[bindx])
            if is_positive:
                next_mask = max_connected_regions(fn_mask[bindx])
                # next_mask = fn_mask[bindx]
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                # loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                loc = num_points - 1
            else:
                next_mask = max_connected_regions(fp_mask[bindx])
                # next_mask = fp_mask[bindx]
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
            
            mask_ind = np.argwhere(next_mask==True)
        if len(mask_ind) > 0:
            y0, y1, x0, x1 = mask_ind[:,0].min(), mask_ind[:,0].max(), mask_ind[:,1].min(), mask_ind[:,1].max()
            
            # jitter_ratio: 0.02
            if jitter_box:
                offset = random.randint(-set_offset, 0)
                begin_x = np.minimum(np.maximum(x0 + offset, 0), mask_width-set_offset)
                offset = random.randint(0, set_offset)
                end_x = np.maximum(np.minimum(x1 + offset, mask_width), begin_x + set_offset)
                
                offset = random.randint(-set_offset, 0)
                begin_y = np.minimum(np.maximum(y0 + offset, 0), mask_height-set_offset)
                offset = random.randint(0, set_offset)
                end_y = np.maximum(np.minimum(y1 + offset, mask_height), begin_y + set_offset)
                
                y0, y1, x0, x1 = begin_y, end_y, begin_x, end_x
            
            x_center = int(0.5*(x0+x1))
            y_center = int(0.5*(y0+y1))
            b_width = int(x1 - x0)
            b_height = int(y1 - y0)
            try:
                assert x_center >= 1
                assert y_center >= 1
                assert b_width >= 1
                assert b_height >= 1

                # x_center, y_center, b_width, b_height, prompts_index
                label_ind = loc.cpu().numpy() if isinstance(loc, torch.Tensor) else loc
                next_boxes[bindx] = [x_center, y_center, b_width, b_height, label_ind]
            except:
                next_boxes[bindx] = [0, 0, 0, 0, 0]
        else:
            next_boxes[bindx] = [0, 0, 0, 0, 0]
        
    return next_boxes



def bezier_curve(points, bbox=None, num_samples=100, as_inline=False):
    if as_inline:
        degree = points.shape[0] -1
        points = points.transpose((1,0))
        curve = bezier.Curve(points, degree=degree)
        s_vals = np.linspace(0.0, 1.0, num_samples)
        data = curve.evaluate_multi(s_vals)
        data = data.transpose((1,0))
        x_new, y_new = data[:,0], data[:,1]
        x_new = np.clip(x_new, bbox[0], bbox[2]).astype(int)
        y_new = np.clip(y_new, bbox[1], bbox[3]).astype(int)
        data = np.column_stack((x_new, y_new))
        
    else:
        try:
            # 将给定点的坐标分别拆分为 x 和 y 坐标
            x = points[:, 0]
            y = points[:, 1]
            # 使用 make_interp_spline 进行贝塞尔曲线插值
            # spline = make_interp_spline(x, y, bc_type="clamped")
            spline = make_interp_spline(x, y)
            # 生成更加平滑的曲线上的点
            x_new = np.linspace(x.min(), x.max(), num_samples)
            y_new = spline(x_new)
            x_new = np.clip(x_new, bbox[0], bbox[2]).astype(int)
            y_new = np.clip(y_new, bbox[1], bbox[3]).astype(int)
            data = np.column_stack((x_new, y_new))
        except:
            degree = points.shape[0] -1
            points = points.transpose((1,0))
            curve = bezier.Curve(points, degree=degree)
            s_vals = np.linspace(0.0, 1.0, num_samples)
            data = curve.evaluate_multi(s_vals)
            data = data.transpose((1,0))
            x_new, y_new = data[:,0], data[:,1]
            x_new = np.clip(x_new, bbox[0], bbox[2]).astype(int)
            y_new = np.clip(y_new, bbox[1], bbox[3]).astype(int)
            data = np.column_stack((x_new, y_new))
    return data

def max_connected_regions(gt_mask):
    from skimage import measure
    labels = measure.label(gt_mask, connectivity=2)
    if np.max(labels) == 0:
        return labels
    max_num = 0
    for j in range(1, np.max(labels)+1):
        if np.sum(labels==j) > max_num:
            max_num = np.sum(labels==j)
            max_pixel = j
        if np.sum(labels==j)>0.1*np.sum(labels!=0):
            labels[labels==j] = max_pixel 
    labels[labels != max_pixel]=0
    labels[labels == max_pixel]=1
    labels = np.array(labels,dtype=np.int8)
    return labels

def cal_scribble(gt_mask, min_p=3, max_p=10, num_samples = 1000):
    bs_scribbles = []
    bs_bounding_rectangles = []
    for i in range(len(gt_mask)):
        if np.sum(gt_mask[i]) > 0:
            mask = max_connected_regions(gt_mask[i])
            mask_ind_hw = np.argwhere(mask==True)
            num_p = random.randint(min_p, max_p)
            # if len(mask_ind_hw) > 0:
            x0, x1, y0, y1 = mask_ind_hw[:,0].min(), mask_ind_hw[:,0].max(), mask_ind_hw[:,1].min(), mask_ind_hw[:,1].max()
            x_center = int(0.5*(x0+x1))
            y_center = int(0.5*(y0+y1))
            b_width = int(x1 - x0)
            b_height = int(y1 - y0)
            # bounding_rectangle = np.array([[x_center, y_center, b_width, b_height]])
            bounding_rectangle = np.array([[y_center, x_center, b_height, b_width]])
            bbox=[x0, y0, x1, y1]
            
            value = x0
            gap = b_width // num_p
            s_points = []
            for ind in range(num_p):
                if value+gap > value:
                    x_point = random.randint(value, value+gap-1)
                else:
                    x_point = random.randint(value, value+gap)
                x_ind = mask_ind_hw[:,0] == x_point
                if mask_ind_hw[x_ind].shape[0] > 0:
                    y_ind = random.randint(0, mask_ind_hw[x_ind].shape[0]-1)
                    s_points.append(mask_ind_hw[x_ind][y_ind])
                value += gap
                
            points = np.array(s_points)
            if len(points) > 0:
                as_inline = np.random.rand() > 0.5
                # as_inline = False
                # scribbles = bezier_curve(points, num_samples, as_inline=as_inline)
                scribbles = bezier_curve(points, bbox, num_samples, as_inline=as_inline)
                scribbles = scribbles[:,::-1]
            else:
                scribbles = np.zeros([num_samples, 2])
                bounding_rectangle = np.array([[0, 0, 0, 0]])
        else:
            scribbles = np.zeros([num_samples, 2])
            bounding_rectangle = np.array([[0, 0, 0, 0]])
        bs_scribbles.append(np.expand_dims(scribbles, 0))
        bs_bounding_rectangles.append(bounding_rectangle)
        
    bs_scribbles = np.expand_dims(np.concatenate(bs_scribbles, 0), 1)
    bs_bounding_rectangles = np.array(bs_bounding_rectangles)
    # return [np.expand_dims(np.concatenate(bs_scribbles, 0), 1), np.array(bs_bounding_rectangles)]
    return [bs_scribbles, bs_bounding_rectangles]



# def cal_scribble(gt_mask, min_p=3, max_p=10, num_samples=2000):
#     bs_scribbles = []
#     bs_bounding_rectangles = []
#     for i in range(len(gt_mask)):
#         mask = max_connected_regions(gt_mask[i])
#         mask_ind_hw = np.argwhere(mask==True)
#         num_p = random.randint(min_p, max_p)
        
#         y0, y1, x0, x1 = mask_ind_hw[:,0].min(), mask_ind_hw[:,0].max(), mask_ind_hw[:,1].min(), mask_ind_hw[:,1].max()
#         # x0, x1, y0, y1 = mask_ind_hw[:,0].min(), mask_ind_hw[:,0].max(), mask_ind_hw[:,1].min(), mask_ind_hw[:,1].max()
#         x_center = int(0.5*(x0+x1))
#         y_center = int(0.5*(y0+y1))
#         b_width = int(x1 - x0)
#         b_height = int(y1 - y0)
#         bounding_rectangle = np.array([[x_center, y_center, b_width, b_height]])
        
#         value = y0
#         gap = b_height // num_p
#         s_points = []
#         for ind in range(num_p):
#             y_point = random.randint(value, value+gap-1)
#             y_ind = mask_ind_hw[:,0] == y_point
#             x_ind = random.randint(0, mask_ind_hw[y_ind].shape[0]-1)
#             s_points.append(mask_ind_hw[y_ind][x_ind])
#             value += gap
            
#         points = np.array(s_points)
#         as_inline = np.random.rand() > 0.5
#         scribbles = bezier_curve(points, num_samples, as_inline=as_inline)
#         bs_scribbles.append(np.expand_dims(scribbles, 0))
#         bs_bounding_rectangles.append(bounding_rectangle)
        
#     bs_scribbles = np.expand_dims(np.concatenate(bs_scribbles, 0), 1)
#     bs_bounding_rectangles = np.array(bs_bounding_rectangles)
#     # return [np.expand_dims(np.concatenate(bs_scribbles, 0), 1), np.array(bs_bounding_rectangles)]
#     return [bs_scribbles, bs_bounding_rectangles]
