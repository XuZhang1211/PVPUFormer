from functools import lru_cache

import cv2
import numpy as np
import copy
import torch

def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            marker = {
                0: cv2.MARKER_CROSS,
                1: cv2.MARKER_DIAMOND,
                2: cv2.MARKER_STAR,
                3: cv2.MARKER_TRIANGLE_UP
            }[p[2]] if p[2] <= 3 else cv2.MARKER_SQUARE
            image = cv2.drawMarker(image, (int(p[1]), int(p[0])),
                                   color, marker, 4, 1)
        else:
            pradius = radius
            image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(255, 0, 0),
                               neg_color=(0, 255, 0), radius=4, mask_color=(255, 165, 0), maskfill=False, mask_alpha=0.5, vis_edge=True, vis_click=True):
    if maskfill:
        result = draw_mask_on_image(img, mask, mask_color, mask_alpha, vis_edge)
    else:
        result = img.copy()
        if mask is not None:
            palette = get_palette(np.max(mask) + 1)
            rgb_mask = palette[mask.astype(np.uint8)][...,::-1]

            if vis_click:
                mask_region = (mask > 0).astype(np.uint8)
                result = result * (1 - mask_region[:, :, np.newaxis]) + \
                    (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                    alpha * rgb_mask
                result = result.astype(np.uint8)
            else:
                rgb_mask[mask > 0] = (0, 0, 255)
                result[mask > 0] = (0, 0, 0)
                # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)
                result = (result+rgb_mask).astype(np.uint8)
    
    if clicks_list is not None and len(clicks_list) > 0 and vis_click:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result

def _transform_clicks(clicks_list, _roi_image, _object_roi=(0, 448, 0, 448)):
    # _object_roi -> (0, 448, 0, 448)
    # image.shape -> crop_height, crop_width 原始图片高宽
    rmin, rmax, cmin, cmax = _object_roi
    if len(_roi_image.shape)==2:
        crop_height, crop_width = _roi_image.shape
    if len(_roi_image.shape)==3:
        crop_height, crop_width, channel = _roi_image.shape

    for ind in range(clicks_list.shape[1]):
        if clicks_list[0, ind, 2] != -1:
            new_r = crop_height * (clicks_list[0, ind, 0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (clicks_list[0, ind, 1] - cmin) / (cmax - cmin + 1)
            clicks_list[0, ind] = np.array([new_r, new_c, clicks_list[0, ind, 2]])
    return clicks_list

def _transform_box(clicks_list, _roi_image, _object_roi=(0, 448, 0, 448)):
    # _object_roi -> (0, 448, 0, 448)
    # image.shape -> crop_height, crop_width 原始图片高宽
    rmin, rmax, cmin, cmax = _object_roi
    if len(_roi_image.shape)==2:
        crop_height, crop_width = _roi_image.shape
    if len(_roi_image.shape)==3:
        crop_height, crop_width, channel = _roi_image.shape

    if np.sum(clicks_list[0]) != 0:
        x_center, y_center, b_width, b_height, label_ind = clicks_list[0]
        y0 = y_center - b_height//2
        y1 = y_center + b_height//2
        x0 = x_center - b_width//2
        x1 = x_center + b_width//2
        
        new_y0 = crop_height * (y0 - rmin) / (rmax - rmin + 1)
        new_y1 = crop_height * (y1 - rmin) / (rmax - rmin + 1)
        new_x0 = crop_width * (x0 - cmin) / (cmax - cmin + 1)
        new_x1 = crop_width * (x1 - cmin) / (cmax - cmin + 1)
        
        y_center = int(0.5*(new_y0+new_y1))
        x_center = int(0.5*(new_x0+new_x1))
        b_height = int(new_y1 - new_y0)
        b_width = int(new_x1 - new_x0)
        
        clicks_list[0] = np.array([x_center, y_center, b_width, b_height, label_ind])
    return clicks_list

def _transform_scribble(clicks_list, _roi_image, _object_roi=(0, 448, 0, 448)):
    # _object_roi -> (0, 448, 0, 448)
    # image.shape -> crop_height, crop_width 原始图片高宽
    scribbles, bounding_rectangle = clicks_list
    
    rmin, rmax, cmin, cmax = _object_roi
    if len(_roi_image.shape)==2:
        crop_height, crop_width = _roi_image.shape
    if len(_roi_image.shape)==3:
        crop_height, crop_width, channel = _roi_image.shape

    for ind in range(scribbles.shape[2]):
        if np.sum(scribbles[0, 0, ind]) != 0:
            new_c = crop_width * (scribbles[0, 0, ind, 0] - cmin) / (cmax - cmin + 1)
            new_r = crop_height * (scribbles[0, 0, ind, 1] - rmin) / (rmax - rmin + 1)
            scribbles[0, 0, ind] = np.array([new_c, new_r])
    return [scribbles, bounding_rectangle]
    
def draw_with_error(mask=None, alpha=0.6, clicks_list=None, prompts=None, pos_color=(255, 0, 0),
                               neg_color=(0, 255, 0), radius=4, mask_color=(255, 165, 0), maskfill=False, mask_alpha=0.5, vis_edge=True, vis_click=True, as_prompt_type=0):
    points, boxes, scribbles = prompts
    points = points.cpu().numpy()
    boxes = boxes.cpu().numpy().astype('int32')
    points = _transform_clicks(points, mask, _object_roi=(0, 448, 0, 448))
    boxes = _transform_box(boxes, mask, _object_roi=(0, 448, 0, 448))
    scribbles = _transform_scribble(scribbles, mask, _object_roi=(0, 448, 0, 448))
    result = mask.copy()
    if clicks_list is not None and len(clicks_list) > 0 and vis_click:
        
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]
        if as_prompt_type == 0:
            result = draw_points(result, pos_points, pos_color, radius=radius)
            result = draw_points(result, neg_points, neg_color, radius=radius)
        
        if as_prompt_type == 1:
            # pos_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2) if points[0][v][2] >= 0]
            # neg_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2,points.shape[1]) if points[0][v][2] >= 0]
        
            # result = draw_points(result, pos_points, pos_color, radius=radius)
            # result = draw_points(result, neg_points, neg_color, radius=radius)
            result = draw_boxs(result, boxes, points, pos_color, radius=radius)
        if as_prompt_type == 2:
            # pos_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2) if points[0][v][2] >= 0]
            # neg_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2,points.shape[1]) if points[0][v][2] >= 0]
            # result = draw_points(result, pos_points, pos_color, radius=radius)
            # result = draw_points(result, neg_points, neg_color, radius=radius)
            result = draw_scribbles(result, scribbles, neg_color, radius=radius)

    return result

def draw_with_blend_and_prompts(img, mask=None, alpha=0.6, clicks_list=None, prompts=None, pos_color=(255, 0, 0),
                               neg_color=(0, 255, 0), radius=4, mask_color=(255, 165, 0), maskfill=False, mask_alpha=0.5, vis_edge=True, vis_click=True, as_prompt_type=0):
    points, boxes, scribbles = prompts
    points = points.cpu().numpy()
    boxes = boxes.cpu().numpy().astype('int32')
    points = _transform_clicks(points, mask, _object_roi=(0, 448, 0, 448))
    boxes = _transform_box(boxes, mask, _object_roi=(0, 448, 0, 448))
    scribbles = _transform_scribble(scribbles, mask, _object_roi=(0, 448, 0, 448))
    if maskfill:
        result = draw_mask_on_image(img, mask, mask_color, mask_alpha, vis_edge)
    else:
        result = img.copy()
        if mask is not None:
            palette = get_palette(np.max(mask) + 1)
            rgb_mask = palette[mask.astype(np.uint8)][...,::-1]

            if vis_click:
                mask_region = (mask > 0).astype(np.uint8)
                result = result * (1 - mask_region[:, :, np.newaxis]) + \
                    (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                    alpha * rgb_mask
                result = result.astype(np.uint8)
            else:
                rgb_mask[mask > 0] = (0, 0, 255)
                result[mask > 0] = (0, 0, 0)
                # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)
                result = (result+rgb_mask).astype(np.uint8)
    
    if clicks_list is not None and len(clicks_list) > 0 and vis_click:
        
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]
        if as_prompt_type == 0:
            result = draw_points(result, pos_points, pos_color, radius=radius)
            result = draw_points(result, neg_points, neg_color, radius=radius)
        
        if as_prompt_type == 1:
            # pos_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2) if points[0][v][2] >= 0]
            # neg_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2,points.shape[1]) if points[0][v][2] >= 0]
            # result = draw_points(result, pos_points, pos_color, radius=radius)
            # result = draw_points(result, neg_points, neg_color, radius=radius)
            result = draw_boxs(result, boxes, points, pos_color, radius=radius)
        if as_prompt_type == 2:
            # pos_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2) if points[0][v][2] >= 0]
            # neg_points = [(points[0][v][0],points[0][v][1]) for v in range(points.shape[1]//2,points.shape[1]) if points[0][v][2] >= 0]
            # result = draw_points(result, pos_points, pos_color, radius=radius)
            # result = draw_points(result, neg_points, neg_color, radius=radius)
            result = draw_scribbles(result, scribbles, neg_color, radius=radius)

    return result

def draw_boxs(image, boxs, points, color, radius=3):
    num_pos = points.shape[1]//2
    image = image.copy()
    # for box in boxs:
    box = boxs[0]
    x_center, y_center, b_width, b_height, boxes_index = box
    
    x0,x1,y0,y1 = x_center-b_width//2, x_center+b_width//2, y_center-b_height//2, y_center+b_height//2
    # image = np.uint8((image.astype(int))*255)
    if boxes_index < num_pos:
        cv2.rectangle(image, (x0, y0), (x1,y1),  (192, 0, 0), 3) # rgb(192, 0, 0)  c00000
    else:
        cv2.rectangle(image, (x0, y0), (x1,y1),  (4, 136, 136), 3) # rgb(4, 136, 136) 048888
    # cv2.rectangle(image, (x0, y0), (x1,y1),  (255, 255, 255), 3)
    return image

def draw_scribbles(image, scribbles, color, radius=3):
    scribble, bounding_rectangle = scribbles
    scribble = scribble[0][0]
    image = image.copy()
    curve = np.column_stack((scribble[:,0].astype(np.int32),scribble[:,1].astype(np.int32)))
    # image = cv2.polylines(image, [curve], False, (255,255,255), 3)
    image = cv2.polylines(image, [curve], False, (192,0,0), 3)  # rgb(192, 0, 0)  c00000

    return image

def draw_mask_on_image(image, mask, color=(255, 165, 0), mask_alpha=0.5, vis_edge=True):
    mask_ind = mask > 0
    mask = np.uint8(mask.astype(int)*255)
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # 绘制mask
    zeros = np.zeros((image.shape), dtype=np.uint8)
    # 原本thickness = -1表示内部填充
    # mask = cv2.fillPoly(zeros, contours, color=(255, 255, 0))
    mask = cv2.fillPoly(zeros, contours, color=color)
    
    if vis_edge:
        image_ = image.copy()
        image_[mask_ind] *= (1 - mask_alpha)
        image = mask_alpha*mask + image_
        cv2.polylines(image, contours, isClosed=True, thickness=3, color=(255, 255, 255))
    else:
        # mask_ = copy.deepcopy(mask)
        # image_ = copy.deepcopy(image)
        mask_ = mask.copy()
        image_ = image.copy()
        # mask_[mask_ind] = (0, 0, int(255 * mask_alpha))
        mask_[mask_ind] = np.array(color)*mask_alpha
        image_[mask_ind] *= (1 - mask_alpha)
        # image[mask_ind] = (0, 0, 0)
        image = mask_ + image_
    return image


def draw_heatmap(img, mask=None, alpha=0.5):
    result = img.copy()
    if mask is not None:
        heatmap = np.uint8(mask.astype(int)*255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        result = heatmap*alpha + img
        
        result = result.astype(np.uint8)
    return result

def draw_clicks(img, clicks_list=None, pos_color=(255, 0, 0),
                               neg_color=(0, 255, 0), radius=4):
    
    result = img.astype(np.uint8)
    
    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result
