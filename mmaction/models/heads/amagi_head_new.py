# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class AmagiHead2(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 channel_rate=2,            ###
                 time_rate=2,               ###
                 speed_ratio=8,             ###
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(2048, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        
        # segmentation
        self.fast_len, self.slow_len, self.time_rate = 8//speed_ratio, 8, time_rate
        self.channel_rate = channel_rate
        self.f_channel = 2048 // self.channel_rate
        self.s_channel = 256 // self.channel_rate

        self.sigmoid = nn.Sigmoid()
        self.seg_reduction = nn.Conv2d(2, 2, 8, stride=8)
        self.fast_conv = nn.Conv3d(2048, self.f_channel, 1, stride=(self.time_rate,1,1))   # 2k-1??
        self.slow_conv = nn.Conv3d(256, self.s_channel, 1, stride=(self.time_rate,1,1))

        self.map_fast = nn.Conv3d(self.f_channel, self.f_channel, 1)
        self.map_slow = nn.Conv3d(self.s_channel, self.s_channel, 1)
        self.map_fast2 = nn.Conv3d(self.f_channel, self.s_channel, (self.fast_len//self.time_rate,1,1))
        self.map_slow2 = nn.Conv3d(self.s_channel, self.s_channel, (8//self.time_rate,1,1))
        self.last_fusion = nn.Conv2d(self.s_channel*2, 2048, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        out, seg = x
        x_fast, x_slow = out

        x_fast = self.fast_conv(x_fast)
        x_slow = self.slow_conv(x_slow)
        #
        seg = seg[0] + seg[1]
        #------------
        #linear mapping
        seg = self.seg_reduction(seg)
        seg = self.sigmoid(seg)
        #------------
        s0, s1 = seg[:,0], seg[:,1]
        
        # mask-weighted fast stream
        seg0 = s0.unsqueeze(1).unsqueeze(1)
        seg0_f = seg0.expand(-1, self.f_channel, self.fast_len//self.time_rate, 7, 7)
        seg0_fast = seg0_f * x_fast    # class0_fast
        seg0_s = seg0.expand(-1, self.s_channel, self.slow_len//self.time_rate, 7, 7)
        seg0_slow = seg0_s * x_slow    # class0_slow

        seg1 = s1.unsqueeze(1).unsqueeze(1)
        seg1_fast = seg1.expand(-1, self.f_channel, self.fast_len//self.time_rate, 7, 7)
        seg1_fast = seg1_fast * x_fast    # class1_fast
        seg1_slow = seg1.expand(-1, self.s_channel, self.slow_len//self.time_rate, 7, 7)
        seg1_slow = seg1_slow * x_slow    # class1_slow

        # combine slows and fast features each, and apply 1*1*1 conv
        fast_feat = torch.add(seg0_fast, seg1_fast)
        slow_feat = torch.add(seg0_slow, seg1_slow)
        fast_feat = self.map_fast(fast_feat)
        slow_feat = self.map_slow(slow_feat)

        # 
        fast_feat = fast_feat.add(x_fast)  # residual connection
        slow_feat = slow_feat.add(x_slow)  # residual connection
        fast_feat = self.map_fast2(fast_feat) # reduce time axis
        slow_feat = self.map_slow2(slow_feat) # reduce time axis

        combined = torch.cat([fast_feat, slow_feat], dim=1).squeeze(2)
        combined = self.last_fusion(combined)
        x = self.gap(combined)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N x C]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        cls_score = self.fc_cls(x)


        return cls_score
