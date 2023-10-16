from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional as F
from torch import batch_norm, einsum
from einops import rearrange, repeat

from v2xvit.models.sub_modules.torch_transformation_utils import warp_affine_simple
from v2xvit.models.comm_modules.comm_module import Communication
from v2xvit.models.fuse_modules.scope_deformable_transformer import RPN_transformer_deformable_mtf_singlescale


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]
        return x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


class SCOPE(nn.Module):
    def __init__(self, args):
        super(SCOPE, self).__init__()
        
        self.max_cav = 5
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0] 
        self.downsample_rate = args['downsample_rate']
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums'] 
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == 'MAX':
                    fuse_network = MaxFusion()
                elif self.agg_mode == "Deform":
                    fuse_network = RPN_transformer_deformable_mtf_singlescale(channel=num_filters[idx])  
                self.fuse_modules.append(fuse_network)
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'MAX':
                self.fuse_modules = MaxFusion()   

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None):
        
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if self.multi_scale:
            ups = []
            with_resnet = True if hasattr(backbone, 'resnet') else False  # True
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                batch_confidence_maps = self.regroup(rm, record_len)

                if i==0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len)
                        _, communication_masks, communication_rates, deform_map = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        x = x * communication_masks
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                batch_node_features = self.regroup(x, record_len)
                
                x_fuse = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features,
                                                    t_matrix[0, :, :, :],
                                                    (H, W))
                    if self.agg_mode == "Deform":
                        feature_shape = neighbor_feature.shape
                        padding_len = self.max_cav - feature_shape[0]
                        padding_feature = torch.zeros(padding_len, feature_shape[1],
                                                    feature_shape[2], feature_shape[3])
                        padding_feature = padding_feature.to(neighbor_feature.device)
                        neighbor_feature = torch.cat([neighbor_feature, padding_feature],
                                                dim=0)
                        if i == 0:
                            padding_map = torch.zeros(padding_len, 1,feature_shape[2], feature_shape[3])
                            padding_map = padding_map.to(neighbor_feature.device)
                            deform_map_b = torch.cat([deform_map[b], padding_map],dim=0)
                        x_fuse.append(self.fuse_modules[i](neighbor_feature, deform_map_b,i))
                    else:
                        x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            batch_node_features = self.regroup(x, record_len)
            batch_confidence_maps = self.regroup(rm, record_len)

            if self.communication:
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else: 
                communication_rates = torch.tensor(0).to(x.device)
            
            x_fuse = []
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse, communication_rates, {}