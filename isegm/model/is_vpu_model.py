import math
import torch
import torch.nn as nn
import numpy as np
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .modeling.detr_transformer import DetrTransformer
from .modeling.transformer import TransformerDecoder, TwoWayTransformer
from .modeling.common import FFNBlock
from typing import Any, Optional, Tuple, Type
from isegm.model.ops import GaussianVector, GaussianVector_box, GaussianVector_scribble
from easydict import EasyDict as edict
 


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024], img_size=(448, 448), decoder_type=''):
        super().__init__()
        
        self.heads=4
        self.dropout=0.1
        self.d_model=in_dim
        self.hide_dim=1024
        # self.depth=1
        self.activation="gelu"
        self.img_size = img_size
        self.return_intermediate = False
        self.decoder_type = decoder_type
        
        
        # self.attn = MultiHeadedAttention(self.heads, self.d_model)
        # self.norm_2 = nn.LayerNorm(self.d_model)
        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)
        self.dropout_3 = nn.Dropout(self.dropout)
        
        self.ffn_layer = FFNBlock(
            embedding_dim=self.img_size[0]*2+3,
            mlp_dim=self.hide_dim*2,
            out_dim=self.d_model,
            )
        
        self.att = TwoWayTransformer(
            depth=3,
            embedding_dim=self.d_model,
            num_heads=8,
            mlp_dim=self.hide_dim,
            activation=nn.ReLU,
            attention_downsample_rate=2,
            return_intermediate=True,
        )
        
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x, q=None, image_position=None, grid_size=None, fusion_type="adaptive_wise", residual=False, edloss=False):
        # import pudb;pudb.set_trace()
        if q.shape[2] != x.shape[2]:
            q = self.ffn_layer(q.type_as(x))
        B, N, C = x.shape
        
        hs = self.att(q, x) # forward(self, vis, txt, vis_pos=None, txt_pos=None, pad_mask=None)
        x2, x3, x4 = hs
        q_x2, x2_q = x2
        q_x3, x3_q = x3
        q_x4, x4_q = x4
        q_out = q + q_x2 + q_x3 + q_x4
        
        q_x2 = q_x2.max(dim=1).values.sigmoid() # global pool without cls token
        q_x3 = q_x3.max(dim=1).values.sigmoid()
        q_x4 = q_x4.max(dim=1).values.sigmoid()
        q_x2 = x * q_x2.unsqueeze(1)
        q_x3 = x * q_x3.unsqueeze(1)
        q_x4 = x * q_x4.unsqueeze(1)
        
        x2_q = x2_q.max(dim=2).values.sigmoid() # global pool without cls token
        x3_q = x3_q.max(dim=2).values.sigmoid()
        x4_q = x4_q.max(dim=2).values.sigmoid()
        x2_q = x * x2_q.unsqueeze(2)
        x3_q = x * x3_q.unsqueeze(2)
        x4_q = x * x4_q.unsqueeze(2)
        x2 = x + q_x2 + x2_q
        x3 = x + q_x3 + x3_q
        x4 = x + q_x4 + x4_q

        x = x.transpose(-1,-2).view(B, -1, grid_size[0], grid_size[1])
        x2 = x2.transpose(-1,-2).view(B, -1, grid_size[0], grid_size[1])
        x3 = x3.transpose(-1,-2).view(B, -1, grid_size[0], grid_size[1])
        x4 = x4.transpose(-1,-2).view(B, -1, grid_size[0], grid_size[1])
        
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x2)
        x_down_16 = self.down_16(x3)
        x_down_32 = self.down_32(x4)

        if edloss:
            return [x_down_4, x_down_8, x_down_16, x_down_32], q_out
        else:
            return [x_down_4, x_down_8, x_down_16, x_down_32]



class VitMultiGaussianVector_ed_Model(ISModel):
    @serialize
    def __init__(
        self,
        num_max_points=24,
        backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
        residual=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.num_max_points = num_max_points
        
        self.image_size = backbone_params['img_size']
        self.vit_patch_size = backbone_params['patch_size']
        self.image_embedding_size_ = self.image_size[0] // self.vit_patch_size[0]
        self.image_embedding_size = [self.image_embedding_size_, self.image_embedding_size_]
        self.embed_dim = backbone_params['embed_dim']
        self.residual = residual
        self.out_chans = 256

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)
        
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)
        
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, self.embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)
        
        self.points_to_vector = None
        if self.with_aux_output:
            self.head_aux = nn.Conv2d(
                    128, 1, kernel_size=1, stride=1, padding=0, bias=True)
    
    
    def _guassinvector_click(self, points):
        model_cfg = edict()
        # model_cfg.num_lmks = self.num_max_points
        model_cfg.input_shape = self.image_size
        model_cfg.sigma = 3
        model_cfg.heighten_peak = True
        model_cfg.upsampling_scale = 4
        model_cfg.input_over_output_stride = 4
        self.gvector = GaussianVector(model_cfg)
        B, N, _ = points.shape
        num_lmks = N // 2
        points_xy, labels = points[:,:,:2].cpu().numpy(), points[:,:,2]
        
        positive_vector_x, positive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,:num_lmks])
        negtive_vector_x, negtive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,num_lmks:])
        not_point_label = torch.zeros(self.image_size[0]*2+3).to(points.device)
        not_point_label[-1] += 1
        
        positive_point_label = torch.zeros((B, num_lmks, 3))
        negtive_point_label = torch.zeros((B, num_lmks, 3))
        positive_point_label[:,:,0] += 1
        negtive_point_label[:,:,1] += 1
        positive_vector = torch.cat([positive_vector_x, positive_vector_y, positive_point_label], dim=2)
        negtive_vector = torch.cat([negtive_vector_x, negtive_vector_y, negtive_point_label], dim=2)
        # torch.Size([2, 48, 899])
        gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
        gaussainvector[labels == -1] = 0
        gaussainvector[labels == -1] += not_point_label
        
        if num_lmks != self.num_max_points:
            positive_vector = gaussainvector[:,:num_lmks]
            negtive_vector = gaussainvector[:,num_lmks:]
            
            num_add_not_points = (self.num_max_points - num_lmks)
            add_not_points = not_point_label.reshape(1,1,-1).repeat(B, num_add_not_points, 1)
            
            negtive_vector = torch.cat([negtive_vector, add_not_points], dim=1)
            positive_vector = torch.cat([positive_vector, add_not_points], dim=1)
            # torch.Size([2, 48, 899])
            gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
            
        return gaussainvector
    
    
    def _guassinvector_box(self, points, boxes):
        model_cfg = edict()
        # model_cfg.num_lmks = self.num_max_points
        model_cfg.input_shape = self.image_size
        model_cfg.sigma = 3
        model_cfg.heighten_peak = True
        model_cfg.upsampling_scale = 4
        model_cfg.input_over_output_stride = 4
        self.gvector = GaussianVector(model_cfg)
        B, N, _ = points.shape
        num_lmks = N // 2
        points_xy, labels = points[:,:,:2].cpu().numpy(), points[:,:,2]
        
        positive_vector_x, positive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,:num_lmks])
        negtive_vector_x, negtive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,num_lmks:])
        not_point_label = torch.zeros(self.image_size[0]*2+3).to(points.device)
        not_point_label[-1] += 1
        
        positive_point_label = torch.zeros((B, num_lmks, 3))
        negtive_point_label = torch.zeros((B, num_lmks, 3))
        positive_point_label[:,:,0] += 1
        negtive_point_label[:,:,1] += 1
        positive_vector = torch.cat([positive_vector_x, positive_vector_y, positive_point_label], dim=2)
        negtive_vector = torch.cat([negtive_vector_x, negtive_vector_y, negtive_point_label], dim=2)
        # torch.Size([2, 48, 899])
        gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
        gaussainvector[labels == -1] = 0
        gaussainvector[labels == -1] += not_point_label
        
        # box to gaussian vector:
        self.gvector_box = GaussianVector_box(model_cfg)
        # bos shape: B, 5
        # boxes_center_xy: B, 1, 2
        boxes_center_xy, boxes_wh, boxes_index = boxes[:, :2], boxes[:, 2:4], boxes[:,-1]
        boxes_center_xy = boxes_center_xy.unsqueeze(1).cpu().numpy()
        boxes_wh = boxes_wh.unsqueeze(1).cpu().numpy()
        boxes_index = boxes_index.cpu()
        box_point_label = torch.zeros((B, 1, 3))
        box_point_label[:,:,0][boxes_index < num_lmks] += 1
        box_point_label[:,:,1][boxes_index >= num_lmks] += 1
        box_vector_x, box_vector_y = self.gvector_box.transform_box_to_vector(boxes_center_xy, boxes_wh)
        box_vector = torch.cat([box_vector_x, box_vector_y, box_point_label], dim=2).to(points.device)
        
        for bi in range(B):
            gaussainvector[bi][boxes_index[bi]] = box_vector[bi][0]
        
        if num_lmks != self.num_max_points:
            positive_vector = gaussainvector[:,:num_lmks]
            negtive_vector = gaussainvector[:,num_lmks:]
            
            num_add_not_points = (self.num_max_points - num_lmks)
            add_not_points = not_point_label.reshape(1,1,-1).repeat(B, num_add_not_points, 1)
            
            negtive_vector = torch.cat([negtive_vector, add_not_points], dim=1)
            positive_vector = torch.cat([positive_vector, add_not_points], dim=1)
            # torch.Size([2, 48, 899])
            gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
            
        return gaussainvector
       
    
    def _guassinvector_scribble(self, points, scribbles_list):
        model_cfg = edict()
        # model_cfg.num_lmks = self.num_max_points
        model_cfg.input_shape = self.image_size
        model_cfg.sigma = 3
        model_cfg.heighten_peak = True
        model_cfg.upsampling_scale = 4
        model_cfg.input_over_output_stride = 4
        self.gvector = GaussianVector(model_cfg)
        B, N, _ = points.shape
        num_lmks = N // 2
        points_xy, labels = points[:,:,:2].cpu().numpy(), points[:,:,2]
        
        positive_vector_x, positive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,:num_lmks])
        negtive_vector_x, negtive_vector_y = self.gvector.transform_lmks_to_vector(points_xy[:,num_lmks:])
        not_point_label = torch.zeros(self.image_size[0]*2+3).to(points.device)
        not_point_label[-1] += 1
        
        positive_point_label = torch.zeros((B, num_lmks, 3))
        negtive_point_label = torch.zeros((B, num_lmks, 3))
        positive_point_label[:,:,0] += 1
        negtive_point_label[:,:,1] += 1
        positive_vector = torch.cat([positive_vector_x, positive_vector_y, positive_point_label], dim=2)
        negtive_vector = torch.cat([negtive_vector_x, negtive_vector_y, negtive_point_label], dim=2)
        # torch.Size([2, 48, 899])
        gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
        gaussainvector[labels == -1] = 0
        gaussainvector[labels == -1] += not_point_label
        
        # scribble to gaussian vector:
        self.gvector_scribble = GaussianVector_scribble(model_cfg)
        scribbles, bounding_rectangles = scribbles_list[0], scribbles_list[1]
        # scribbles: B, num_sample_points, 2
        # scribbles = scribbles.cpu().numpy()
        # bounding_rectangles = bounding_rectangles.cpu().numpy()
        scribble_index = torch.argwhere(labels[:,:num_lmks] != -1)
        # scribble_index = scribble_index[scribble_index[:,0]==bi][-1][1]
        scribble_label = torch.zeros((B, 1, 3))
        scribble_label[:,:,0] += 1
        scribble_vector_x, scribble_vector_y = self.gvector_scribble.transform_scribble_to_vector(scribbles, bounding_rectangles)
        scribble_vector = torch.cat([scribble_vector_x, scribble_vector_y, scribble_label], dim=2).to(points.device)
        
        for bi in range(B):
            if len(scribble_index[scribble_index[:,0]==bi]) > 0:
                gaussainvector[bi][scribble_index[scribble_index[:,0]==bi][-1][1]] = scribble_vector[bi][0]
        
        if num_lmks != self.num_max_points:
            positive_vector = gaussainvector[:,:num_lmks]
            negtive_vector = gaussainvector[:,num_lmks:]
            
            num_add_not_points = (self.num_max_points - num_lmks)
            add_not_points = not_point_label.reshape(1,1,-1).repeat(B, num_add_not_points, 1)
            
            negtive_vector = torch.cat([negtive_vector, add_not_points], dim=1)
            positive_vector = torch.cat([positive_vector, add_not_points], dim=1)
            # torch.Size([2, 48, 899])
            gaussainvector = torch.cat([positive_vector, negtive_vector], dim=1).to(points.device)
            
        return gaussainvector
       
    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        num_positive = labels.shape[1]//2
        num_negtive = num_positive
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            num_negtive += 1
        labels_p = labels.clone()
        labels_n = labels.clone()
        labels_p[:,num_positive:] = -1
        labels_n[:,:num_positive] = -1
        # torch.Size([2, 48, 768])
        point_embedding = self.pe_layer.forward_with_coords(points, self.image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels_p == 100] += self.point_embeddings[0].weight
        point_embedding[labels_n == 100] += self.point_embeddings[1].weight
        return point_embedding
    

    def backbone_forward(self, image, coord_features=None, points=None, prompts=None, as_prompt_type=0, edloss=True, pclout=False):
        
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, shuffle=self.random_split)
        grid_size = self.backbone.patch_embed.grid_size
        
        # B, N, C = backbone_features.shape
        # image_pe=self.get_dense_pe()
        # image_position = torch.repeat_interleave(image_pe, image.shape[0], dim=0)
        # image_position = image_position.view(B, C, -1).transpose(-1,-2)
        image_position = None
        
        # if prompts is not None:
        if as_prompt_type != 0:
            points, boxes, scribbles = prompts
        if as_prompt_type == 0:
            point_vector = self._guassinvector_click(points)
        elif as_prompt_type == 1:
            point_vector = self._guassinvector_box(points, boxes)
        else:
            point_vector = self._guassinvector_scribble(points, scribbles)
            
        if edloss:
            multi_scale_features, q_out = self.neck(backbone_features, point_vector, image_position, grid_size, residual=self.residual, edloss=True)
            # if pclout:
            output, logits = self.head.forward_feat(multi_scale_features, q_out, pclout=pclout)
            # else:
            #     output = self.head.forward_feat(multi_scale_features, q_out, pclout=pclout)
        else:
            multi_scale_features = self.neck(backbone_features, point_vector, image_position, grid_size, residual=self.residual)
            output = self.head(multi_scale_features)
        
        # if self.with_aux_output and pclout:
        if self.with_aux_output:
            return {'instances': output, 'instances_aux': logits}
        else:
            return {'instances': output, 'instances_aux': None}
    
    
    def forward(self, image, points=None, prompts=None, as_prompt_type=0, edloss=True, pclout=False):
        # if as_prompt_type != 0:
        #     points = torch.zeros_like(points).to(points.device)

        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features_with_prompt(image, prev_mask, points, prompts, as_prompt_type)
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features, points, prompts, as_prompt_type, edloss, pclout)
        # print(outputs['instances'].shape)
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        # if self.with_aux_output and pclout:
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)

        return outputs
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    

    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
