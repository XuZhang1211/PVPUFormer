import sys
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np


sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks, draw_clicks
from isegm.inference.predictors import get_predictor
from isegm.inference.sam_evaluation import evaluate_dataset
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from segment_anything import sam_model_registry, SamPredictor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')
    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,PascalVOC,COCO_MVal,SBD',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.95,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help="Possible choices: cvpr, fixed<number>, or fixed<number>,<number>,(e.g. fixed400, fixed400,600).")

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--cf-n', default=0, type=int,
                        help='cascade-forward step')
    parser.add_argument('--cf-click', default=1, type=int,
                        help='cascade-forward clicks')
    parser.add_argument('--acf', action='store_true', default=False,
                        help='adaptive cascade-forward')
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def main():
    args, cfg = parse_args()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = True
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)

        for checkpoint_path in checkpoints_list:
            # print('Using checkpoint', checkpoint_path)
            # model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm)

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name, eval_ritm=args.eval_ritm)

            # For SimpleClick models, we usually need to interpolate the positional embedding
            # if not args.eval_ritm:
            #     try:
            #         interpolate_pos_embed_inference(model.backbone, zoomin_params['target_size'], args.device)
            #     except:
            #         # interpolate_pos_embed_inference(model.feature_extractor, zoomin_params['target_size'], args.device)
            #         pass
                
            # predictor = get_predictor(model, args.mode, args.device,
            #                           prob_thresh=args.thresh,
            #                           predictor_params=predictor_params,
            #                           zoom_in_params=zoomin_params)
            
            if "vit_b" in checkpoint_path.name:
                model_type = "vit_b"
            if "vit_l" in checkpoint_path.name:
                model_type = "vit_l"
            if "vit_h" in checkpoint_path.name:
                model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=args.device)
            predictor = SamPredictor(sam)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh, args.n_clicks) if args.vis_preds else None
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)

            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)
            print_header = False

    # uncomment the following lines for memory analysis
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(1)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(1)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(1)/1024/1024/1024))

def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):
    predictor_params = {
        'cascade_step': args.cf_n + 1,
        'cascade_adaptive': args.acf,
        'cascade_clicks': args.cf_click
    }

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'target_size': 600 if dataset_name == 'DAVIS' else 400
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size, crop_size)
            }
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448)
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size_h, crop_size_w)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem
        # mid_path = "_".join((args.checkpoint).split('/')[-4:-1])
        # logs_path = args.logs_path / mid_path / checkpoints_list[0].stem

    if args.cf_n > 0:
        cf_prefix = f'acf{args.cf_n}' if args.acf else f'cf{args.cf_n}'
        cf_prefix = f'{cf_prefix}_{args.cf_click}clk'
        if logs_prefix:
            logs_prefix = '_'.join([cf_prefix, logs_prefix])
        else:
            logs_prefix = cf_prefix

    return checkpoints_list, logs_path, logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, noc_list_std, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    # print(noc_list, noc_list_std)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                             for click_id in [_ for _ in range(1, 21)] if click_id <= min_num_clicks])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90, 95]:
            noc_list, _, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious
        }, f)

def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh, max_clicks=2):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    cache = {}
    # import pudb;pudb.set_trace()
    def callback(image, gt_mask, pred_probs, pred_mask, iou,
                 sample_id, click_indx, clicks_list, success,
                 zoom_in=None):

        if cache.get('sample_id') != sample_id or cache.get('click_indx', -1) > click_indx:
            # move to next sample
            cache['sample_id'] = sample_id
            cache['plot'] = None
            cache['iou'] = 0
            cache['click_indx'] = -1

        cache['iou'] = max(iou, cache['iou'])
        cache['click_indx'] = click_indx

        sample_path = save_path / f'{sample_id}.jpg'
        # sample_path = save_path / f'{sample_id}_{click_indx}.jpg'

        # rmin, rmax, cmin, cmax = zoom_in._object_roi
        

        pred_map = pred_probs > prob_thresh
        prob_map = draw_probmap(pred_probs)[..., ::-1]
        # image_with_mask = draw_with_blend_and_clicks(image, pred_map, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(255, 0, 0), maskfill=True, vis_edge=False)
        # image_with_gtmask = draw_with_blend_and_clicks(image, gt_mask, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(255, 0, 0), maskfill=True, vis_edge=False)
        
        image_with_mask = draw_with_blend_and_clicks(image, pred_map, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(255, 165, 0), maskfill=True, vis_edge=True, vis_click=True)
        image_with_gtmask = draw_with_blend_and_clicks(image, gt_mask, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(0, 0, 255), maskfill=True, mask_alpha=0.7, vis_edge=True, vis_click=False)
        image_with_gtmask2 = draw_with_blend_and_clicks(image, gt_mask, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(0, 255, 0), maskfill=True, mask_alpha=0.5, vis_edge=True, vis_click=False)
        
        # pred_mask = pred_map.astype(int)*255
        pred_mask = pred_mask.astype(int)*255
        pred_mask = np.stack((pred_mask,)*3, axis=-1).astype(np.uint8)
        # prob_map = draw_mask_on_image(image, pred_map, (255, 153, 18))
        # prob_map = prob_map*0.9 + image

        error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        error_map[(gt_mask > 0) & ~pred_map] = (255, 0, 0)  # under-segm. fn
        error_map[(gt_mask < 1) & pred_map] = (0, 0, 255)  # over-segm. fp
        
        pred_mask = cv2.putText(pred_mask, f'IoU={iou*100:.2f}%', (0, 70),
                                cv2.FONT_HERSHEY_TRIPLEX, 
                                0.75, (255, 255, 255), 1, cv2.LINE_AA)
        pred_mask = cv2.putText(pred_mask, f'NoC={click_indx+1}', (0, 30),
                                      cv2.FONT_HERSHEY_TRIPLEX, 
                                      0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # image_with_mask = cv2.putText(image_with_mask, f'IoU={iou*100:.2f}%', (0, 70),
        #                         cv2.FONT_HERSHEY_TRIPLEX, 
        #                         0.75, (0, 0, 0), 2, cv2.LINE_AA)
        # image_with_mask = cv2.putText(image_with_mask, f'NoC={click_indx+1}', (0, 30),
        #                               cv2.FONT_HERSHEY_TRIPLEX, 
        #                               0.75, (0, 0, 0), 2, cv2.LINE_AA)
        
        # prob_map = cv2.putText(prob_map.astype(np.uint8), f'IoU={iou*100:.2f}%', (0, 70),
        #                         cv2.FONT_HERSHEY_TRIPLEX, 
        #                         0.75, (255, 255, 255), 1, cv2.LINE_AA)
        # prob_map = cv2.putText(prob_map.astype(np.uint8), f'NoC={click_indx+1}', (0, 30),
        #                               cv2.FONT_HERSHEY_TRIPLEX, 
        #                               0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        error_map = cv2.putText(error_map.astype(np.uint8), f'IoU={iou*100:.2f}%', (0, 70),
                                cv2.FONT_HERSHEY_TRIPLEX, 
                                0.75, (255, 255, 255), 1, cv2.LINE_AA)
        error_map = cv2.putText(error_map.astype(np.uint8), f'NoC={click_indx+1}', (0, 30),
                                      cv2.FONT_HERSHEY_TRIPLEX, 
                                      0.75, (255, 255, 255), 1, cv2.LINE_AA)

        gt_map = gt_mask[..., None].astype(np.uint8)
        gt_map = np.repeat(gt_map, 3, axis=2) * 255
        
        # row1 = np.concatenate((image, gt_map), axis=1)
        # row2 = np.concatenate((image_with_mask, prob_map), axis=1)
        # row3 = np.concatenate((pred_mask, error_map), axis=1)
        # plot = np.concatenate((row1, row2, row3), axis=1)[...,::-1]
        
        row1 = np.concatenate((image_with_gtmask, image_with_gtmask2, image_with_mask), axis=1)
        row2 = np.concatenate((prob_map, pred_mask), axis=1)
        plot = np.concatenate((row1, row2), axis=1)[...,::-1]

        if cache.get('plot', None) is not None:
            plot = np.concatenate((cache['plot'], plot), axis=0)

        cache['plot'] = plot

        if click_indx + 1 == max_clicks and cache['iou'] <= 0.9:
            cv2.imwrite(str(sample_path), plot)
        else:
            cv2.imwrite(str(sample_path), plot)

    return callback


def get_prediction_vis_callback2(logs_path, dataset_name, prob_thresh, max_clicks=2):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    cache = {}
    # import pudb;pudb.set_trace()
    def callback(image, gt_mask, pred_probs, iou,
                 sample_id, click_indx, clicks_list, success,
                 zoom_in):

        if cache.get('sample_id') != sample_id or cache.get('click_indx', -1) > click_indx:
            # move to next sample
            cache['sample_id'] = sample_id
            cache['plot'] = None
            cache['iou'] = 0
            cache['click_indx'] = -1

        cache['iou'] = max(iou, cache['iou'])
        cache['click_indx'] = click_indx

        sample_path = save_path / f'{sample_id}.jpg'
        # sample_path = save_path / f'{sample_id}_{click_indx}.jpg'

        rmin, rmax, cmin, cmax = zoom_in._object_roi
        # import pudb;pudb.set_trace()

        pred_map = pred_probs > prob_thresh
        prob_map = draw_probmap(pred_probs)[..., ::-1]
        image_with_mask = draw_with_blend_and_clicks(image, pred_map, alpha=0.9, clicks_list=clicks_list, radius=7, maskfill=True)
        # image_with_mask = draw_with_blend_and_clicks(image, gt_mask, alpha=0.9, clicks_list=clicks_list, radius=7, mask_color=(255, 0, 0), maskfill=True)
        
        
        pred_mask = pred_map.astype(int)*255
        pred_mask = np.stack((pred_mask,)*3, axis=-1).astype(np.uint8)
        # prob_map = draw_mask_on_image(image, pred_map, (255, 153, 18))
        # prob_map = prob_map*0.9 + image

        error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        error_map[(gt_mask > 0) & ~pred_map] = (255, 0, 0)  # under-segm. fn
        error_map[(gt_mask < 1) & pred_map] = (0, 0, 255)  # over-segm. fp
        
        pred_mask = cv2.putText(pred_mask, f'IoU={iou*100:.2f}%', (0, 70),
                                cv2.FONT_HERSHEY_TRIPLEX, 
                                0.75, (255, 255, 255), 1, cv2.LINE_AA)
        pred_mask = cv2.putText(pred_mask, f'NoC={click_indx+1}', (0, 30),
                                      cv2.FONT_HERSHEY_TRIPLEX, 
                                      0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # image_with_mask = cv2.putText(image_with_mask, f'IoU={iou*100:.2f}%', (0, 70),
        #                         cv2.FONT_HERSHEY_TRIPLEX, 
        #                         0.75, (0, 0, 0), 2, cv2.LINE_AA)
        # image_with_mask = cv2.putText(image_with_mask, f'NoC={click_indx+1}', (0, 30),
        #                               cv2.FONT_HERSHEY_TRIPLEX, 
        #                               0.75, (0, 0, 0), 2, cv2.LINE_AA)
        
        # prob_map = cv2.putText(prob_map.astype(np.uint8), f'IoU={iou*100:.2f}%', (0, 70),
        #                         cv2.FONT_HERSHEY_TRIPLEX, 
        #                         0.75, (255, 255, 255), 1, cv2.LINE_AA)
        # prob_map = cv2.putText(prob_map.astype(np.uint8), f'NoC={click_indx+1}', (0, 30),
        #                               cv2.FONT_HERSHEY_TRIPLEX, 
        #                               0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # error_map = cv2.putText(error_map.astype(np.uint8), f'IoU={iou*100:.2f}%', (0, 70),
        #                         cv2.FONT_HERSHEY_TRIPLEX, 
        #                         0.75, (255, 255, 255), 1, cv2.LINE_AA)
        # error_map = cv2.putText(error_map.astype(np.uint8), f'NoC={click_indx+1}', (0, 30),
        #                               cv2.FONT_HERSHEY_TRIPLEX, 
        #                               0.75, (255, 255, 255), 1, cv2.LINE_AA)

        gt_map = gt_mask[..., None].astype(np.uint8)
        gt_map = np.repeat(gt_map, 3, axis=2) * 255
        
        row1 = np.concatenate((image, gt_map), axis=1)
        row2 = np.concatenate((image_with_mask, prob_map), axis=1)
        row3 = np.concatenate((pred_mask, error_map), axis=1)
        plot = np.concatenate((row1, row2, row3), axis=1)[...,::-1]

        if cache.get('plot', None) is not None:
            plot = np.concatenate((cache['plot'], plot), axis=0)

        cache['plot'] = plot

        if click_indx + 1 == max_clicks and cache['iou'] <= 0.9:
            cv2.imwrite(str(sample_path), plot)
        else:
            cv2.imwrite(str(sample_path), plot)

    return callback


if __name__ == '__main__':
    main()
