import torch
import random
import cv2
from torch import nn as nn
import numpy as np
import isegm.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class GaussianVector(nn.Module):
    def __init__(self, config):
        super(GaussianVector, self).__init__()
        self.config = config
        # self.num_lmks = config.num_lmks # 68
        self.input_h = config.input_shape[0] # 64
        self.input_w = config.input_shape[1] # 64
        self.upscale = config.get('upsampling_scale', 4)
        self.stride = config.input_over_output_stride
        self.output_h = int(self.input_h * self.upscale / self.stride) # 256
        self.output_w = int(self.input_w * self.upscale / self.stride) # 256

        gaussian_sigma = config.sigma
        self.gaussian_radius = int(gaussian_sigma * 3)
        gaussian_kernel_size = 2 * self.gaussian_radius + 1
        gaussian_center = gaussian_kernel_size // 2
        self.gaussian_clip = np.arange(0, gaussian_kernel_size, 1, np.float32)
        self.gaussian_clip = np.exp(-((self.gaussian_clip -
                                       gaussian_center) ** 2) / (2 * gaussian_sigma ** 2))  # noqa

        self.heighten_peak = config.get('heighten_peak', False)
        if self.heighten_peak:
            self.gaussian_clip[gaussian_center] += 1
    
    def is_point_in_img(self, x, y, w, h):
        if (x < 0) or (x > w) or (y < 0) or (y > h):
            return False
        else:
            return True

    def transform_lmks_to_vector(self, lmks):
        B, N, _ = lmks.shape
        vector_x = np.zeros((B, N, self.output_w))
        vector_y = np.zeros((B, N, self.output_h))
        for j in range(B):
            for i in range(N):
                vector_x[j][i], vector_y[j][i] = self.gen_guassian_vector(lmks[j][i])
        vector_x = torch.tensor(vector_x)
        vector_y = torch.tensor(vector_y)
        return vector_x, vector_y

    def gen_guassian_vector(self, lmks):
        lmks = (lmks * self.upscale / self.stride).astype('int32')
        x, y = lmks[0], lmks[1]
        vector_x_instance = np.zeros((self.output_w))
        vector_y_instance = np.zeros((self.output_h))
        ul = [int(x - self.gaussian_radius),
              int(y - self.gaussian_radius)]
        br = [int(x + self.gaussian_radius + 1),
              int(y + self.gaussian_radius + 1)]

        if (not self.is_point_in_img(ul[0], ul[1],
                                self.output_w, self.output_h)) and \
                (not self.is_point_in_img(br[0], br[1],
                                     self.output_w, self.output_h)):
            return vector_x_instance, vector_y_instance

        g_x = max(0, -ul[0]), min(self.output_w, br[0]) - ul[0]
        g_y = max(0, -ul[1]), min(self.output_h, br[1]) - ul[1]
        img_x = max(0, ul[0]), min(self.output_w, br[0])
        img_y = max(0, ul[1]), min(self.output_h, br[1])
        vector_x_instance[img_x[0]:img_x[1]] = \
            self.gaussian_clip[g_x[0]:g_x[1]]
        vector_y_instance[img_y[0]:img_y[1]] = \
            self.gaussian_clip[g_y[0]:g_y[1]]
        return vector_x_instance, vector_y_instance
   

class GaussianVector_box(nn.Module):
    def __init__(self, config):
        super(GaussianVector_box, self).__init__()
        self.config = config
        # self.num_lmks = config.num_lmks # 68
        self.input_h = config.input_shape[0] # 64
        self.input_w = config.input_shape[1] # 64
        self.upscale = config.get('upsampling_scale', 4)
        self.stride = config.input_over_output_stride
        self.output_h = int(self.input_h * self.upscale / self.stride) # 256
        self.output_w = int(self.input_w * self.upscale / self.stride) # 256

        self.gaussian_sigma = config.sigma

    def is_point_in_img(self, x, y, w, h):
        if (x < 0) or (x > w) or (y < 0) or (y > h):
            return False
        else:
            return True

    def transform_box_to_vector(self, boxes_center, boxes_wh):
        B, N, _ = boxes_center.shape
        vector_x = np.zeros((B, N, self.output_w))
        vector_y = np.zeros((B, N, self.output_h))
        for j in range(B):
            for i in range(N):
                vector_x[j][i], vector_y[j][i] = self.gen_guassian_vector(boxes_center[j][i], boxes_wh[j][i])
        vector_x = torch.tensor(vector_x)
        vector_y = torch.tensor(vector_y)
        return vector_x, vector_y

    def gen_guassian_vector(self, lmks, wh):
        vector_x_instance = np.zeros((self.output_w))
        vector_y_instance = np.zeros((self.output_h))
        
        if np.sum(lmks)+np.sum(wh) == 0:
            return vector_x_instance, vector_y_instance
        W, H = wh[0], wh[1]
        # self.gaussian_radius_w = int(self.gaussian_sigma * 3)
        # gaussian_kernel_size_w = 2 * self.gaussian_radius_w + 1
        gaussian_kernel_size_w = W // 2 * 2 - 1
        self.gaussian_radius_w = (gaussian_kernel_size_w - 1) // 2
        self.gaussian_sigma_w = self.gaussian_radius_w // 3
        if self.gaussian_sigma_w == 0:
            return vector_x_instance, vector_y_instance
        gaussian_center_w = gaussian_kernel_size_w // 2
        self.gaussian_clip_w = np.arange(0, gaussian_kernel_size_w, 1, np.float32)
        self.gaussian_clip_w = np.exp(-((self.gaussian_clip_w -
                                       gaussian_center_w) ** 2) / (2 * self.gaussian_sigma_w ** 2))  # noqa
        
        # self.gaussian_radius_h = int(self.gaussian_sigma * 3)
        # gaussian_kernel_size_h = 2 * self.gaussian_radius_h + 1
        gaussian_kernel_size_h = H // 2 * 2 - 1
        self.gaussian_radius_h = (gaussian_kernel_size_h - 1) // 2
        self.gaussian_sigma_h = self.gaussian_radius_h // 3
        if self.gaussian_sigma_h == 0:
            return vector_x_instance, vector_y_instance
        gaussian_center_h = gaussian_kernel_size_h // 2
        self.gaussian_clip_h = np.arange(0, gaussian_kernel_size_h, 1, np.float32)
        self.gaussian_clip_h = np.exp(-((self.gaussian_clip_h -
                                       gaussian_center_h) ** 2) / (2 * self.gaussian_sigma_h ** 2))  # noqa
    
    
        lmks = (lmks * self.upscale / self.stride).astype('int32')
        x, y = lmks[0], lmks[1]
        
        ul = [int(x - self.gaussian_radius_w),
              int(y - self.gaussian_radius_h)]
        br = [int(x + self.gaussian_radius_w + 1),
              int(y + self.gaussian_radius_h + 1)]

        if (not self.is_point_in_img(ul[0], ul[1],
                                self.output_w, self.output_h)) and \
                (not self.is_point_in_img(br[0], br[1],
                                     self.output_w, self.output_h)):
            return vector_x_instance, vector_y_instance
        try:
            g_x = max(0, -ul[0]), min(self.output_w, br[0]) - ul[0]
            g_y = max(0, -ul[1]), min(self.output_h, br[1]) - ul[1]
            img_x = max(0, ul[0]), min(self.output_w, br[0])
            img_y = max(0, ul[1]), min(self.output_h, br[1])
            vector_x_instance[img_x[0]:img_x[1]] = \
                self.gaussian_clip_w[g_x[0]:g_x[1]]
            vector_y_instance[img_y[0]:img_y[1]] = \
                self.gaussian_clip_h[g_y[0]:g_y[1]]
        except:
            import pudb;pudb.set_trace()
            g_x = max(0, -ul[0]), min(self.output_w, br[0]) - ul[0]
            g_y = max(0, -ul[1]), min(self.output_h, br[1]) - ul[1]
            img_x = max(0, ul[0]), min(self.output_w, br[0])
            img_y = max(0, ul[1]), min(self.output_h, br[1])
            vector_x_instance[img_x[0]:img_x[1]] = \
                self.gaussian_clip_w[g_x[0]:g_x[1]]
            vector_y_instance[img_y[0]:img_y[1]] = \
                self.gaussian_clip_h[g_y[0]:g_y[1]]
        return vector_x_instance, vector_y_instance
    

class GaussianVector_scribble(nn.Module):
    def __init__(self, config):
        super(GaussianVector_scribble, self).__init__()
        self.config = config
        # self.num_lmks = config.num_lmks # 68
        self.input_h = config.input_shape[0] # 64
        self.input_w = config.input_shape[1] # 64
        self.upscale = config.get('upsampling_scale', 4)
        self.stride = config.input_over_output_stride
        self.output_h = int(self.input_h * self.upscale / self.stride) # 256
        self.output_w = int(self.input_w * self.upscale / self.stride) # 256

        self.gaussian_sigma = config.sigma

    def is_point_in_img(self, x, y, w, h):
        if (x < 0) or (x > w) or (y < 0) or (y > h):
            return False
        else:
            return True

    def transform_scribble_to_vector(self, scribbles, bounding_rectangles):
        scribbles =  scribbles.astype(np.int32)
        # bounding_rectangles: B, N, 4
        B, N, _, _ = scribbles.shape
        vector_x = np.zeros((B, N, self.output_w))
        vector_y = np.zeros((B, N, self.output_h))
        for j in range(B):
            for i in range(N):
                vector_x[j][i], vector_y[j][i] = self.gen_guassian_vector(scribbles[j][i], bounding_rectangles[j][i])
        vector_x = torch.tensor(vector_x)
        vector_y = torch.tensor(vector_y)
        return vector_x, vector_y

    def drop_overlap_points(self, points):
        points_set = set()
        for point in points:
            points_set.add(tuple(point))
        unique_points = np.array(list(points_set))
        return unique_points
    
    def gen_guassian_vector(self, scribble, bounding_rectangle):
        vector_x_instance = np.zeros((self.output_w))
        vector_y_instance = np.zeros((self.output_h))
        
        if np.sum(scribble) + np.sum(bounding_rectangle) == 0:
            return vector_x_instance, vector_y_instance
        draw_scribble(scribble, bounding_rectangle)
        # drop duplicate points
        # scribble = self.drop_overlap_points(scribble)
        
        scribble = (scribble * self.upscale / self.stride).astype('int32')
        self.gaussian_radius = int(self.gaussian_sigma * 3)
        gaussian_kernel_size = 2 * self.gaussian_radius + 1
        gaussian_center = gaussian_kernel_size // 2
        
        num_points, _ = scribble.shape
        x0, y0, w0, h0 = bounding_rectangle
        x0 = min(x0, self.input_w)
        y0 = min(y0, self.input_h)
        w0 = min(w0, self.input_w)
        h0 = min(h0, self.input_h)
        
        w_yj_box = x0 - w0//2
        h_xi_box = y0 - h0//2
        h_points = []
        
        for xi in range(w0):
            indxs = np.argwhere(scribble[:,0] == xi)
            if len(indxs) != 0:
                indx = random.randint(0, len(indxs)-1)
                point = scribble[indx]
                xi_s, h_xi_s = point
                q_h_xi = np.exp(-((h_xi_s - h_xi_box) ** 2) / (2 * self.gaussian_sigma ** 2))
                vector_x_instance[xi] = q_h_xi
                h_points.append(point)
                
                # drop the selected points 
                ind_xi_s = scribble[:,0] == xi_s
                ind_h_xi_s = scribble[:,1] == h_xi_s
                drop_indxs = np.argwhere(ind_xi_s & ind_h_xi_s)
                scribble = np.delete(scribble, drop_indxs, axis=0)
            
        for yj in range(h0):
            indxs = np.argwhere(scribble[:,1] == yj)
            if len(indxs) != 0:
                indx = random.randint(0, len(indxs)-1)
                point = scribble[indx]
                w_yj_s, h_yj_s = point
                q_w_yj = np.exp(-((w_yj_s - w_yj_box) ** 2) / (2 * self.gaussian_sigma ** 2))
                vector_y_instance[yj] = q_w_yj
                
        # # ----
        # # scribble: (x, y)
        # for xi in range(w0):
        #     indxs = np.argwhere(scribble[:,0] == xi)
        #     if len(indxs) != 0:
        #         indx = random.randint(0, len(indxs)-1)
        #         point = scribble[indx]
        #         xi_s, h_xi_s = point
        #         q_h_xi = np.exp(-((h_xi_s - h_xi_box) ** 2) / (2 * self.gaussian_sigma ** 2))
        #         vector_x_instance[xi] = q_h_xi
        #         h_points.append(point)
                
        #         # drop the selected points 
        #         ind_xi_s = scribble[:,0] == xi_s
        #         ind_h_xi_s = scribble[:,1] == h_xi_s
        #         drop_indxs = np.argwhere(ind_xi_s & ind_h_xi_s)
        #         scribble = np.delete(scribble, drop_indxs, axis=0)
        
        # # scribble: (y, x)
        # for yj in range(h0):
        #     indxs = np.argwhere(scribble[:,1] == yj)
        #     if len(indxs) != 0:
        #         indx = random.randint(0, len(indxs)-1)
        #         point = scribble[indx]
        #         w_yj_s, h_yj_s = point
        #         q_w_yj = np.exp(-((w_yj_s - w_yj_box) ** 2) / (2 * self.gaussian_sigma ** 2))
        #         vector_y_instance[yj] = q_w_yj
                
            
        return vector_x_instance, vector_y_instance
  
    
class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * 2),num_points, x 1 x h x w -> (bs * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()
        # coords shape -> 448,448
        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor
    
def draw_scribble(scribble, bounding_rectangle):
    import cv2
    image = np.zeros([488,488])
    x_center, y_center, b_width, b_height = bounding_rectangle
    x0,x1,y0,y1 = x_center-b_width//2, x_center+b_width//2, y_center-b_height//2, y_center+b_height//2
    image = np.uint8((image.astype(int))*255)
    cv2.rectangle(image, (x_center-b_width//2, y_center-b_height//2), (x_center+b_width//2,y_center+b_height//2),  (255, 255, 255), 3)
    
    curve = np.column_stack((scribble[:,0].astype(np.int32),scribble[:,1].astype(np.int32)))
    image = cv2.polylines(image, [curve], False, (255,255,255), 3)
    cv2.imwrite('/data1/zx/work/ClickSeg/CFR_ICL/results/plt.jpg',image)