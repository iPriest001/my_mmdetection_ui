import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio, K, kernel_size, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        self.ks = kernel_size
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            # nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )
        self.n_fc = nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)  # n*1
        self.cin_fc = nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)  # n*1
        self.cin_sigmoid = nn.Sigmoid()
        self.k2_fc = nn.Conv2d(hidden_planes, kernel_size*kernel_size, kernel_size=1, bias=False)  # n*1
        self.k2_sigmoid = nn.Sigmoid()
        self.out_fc = nn.Conv2d(hidden_planes, out_planes, kernel_size=1, bias=False)
        self.out_sigmoid = nn.Sigmoid()

        if (init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if (self.temprature > 1):
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att)  # bs,hidden_dim,1,1
        n_att = F.softmax(self.n_fc(att).view(x.shape[0], -1) / self.temprature, -1)  # bs, K
        cin_att = self.cin_sigmoid(self.cin_fc(att).view(x.shape[0], 1, -1, 1, 1)) # bs, Cin, 1, 1
        k2_att = self.k2_sigmoid(self.k2_fc(att).view(x.shape[0], 1, 1, self.ks, self.ks)) # bs, ks, ks
        out_att = self.out_sigmoid(self.out_fc(att).view(x.shape[0], -1, 1, 1, 1))
        return [n_att, cin_att, k2_att, out_att]


class AdaptiveAngleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, angle_list=[0]):
        super(AdaptiveAngleConv, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.angle_list = angle_list
        self.branches = len(angle_list)
        self.attention = Attention(in_planes=in_channel, out_planes=out_channel, ratio=16, K=len(angle_list), kernel_size=kernel_size, temprature=30, init_weight=True)
        self.baseline_conv = nn.Conv2d(in_channel, out_channel, (kernel_size, kernel_size), (stride, stride), (padding, padding))
        self.bias = nn.Parameter(torch.randn(len(angle_list)-1, out_channel), requires_grad=True)
        self.K = len(angle_list)

    def forward(self, x):
        k1 = self._rotate_kernel(self.baseline_conv.weight, self.angle_list[1])
        k2 = self._rotate_kernel(self.baseline_conv.weight, self.angle_list[2])
        k3 = self._rotate_kernel(self.baseline_conv.weight, self.angle_list[3])
        k4 = self._rotate_kernel(self.baseline_conv.weight, self.angle_list[4])
        k_sum = torch.cat([self.baseline_conv.weight.unsqueeze(0), k1.unsqueeze(0), k2.unsqueeze(0), k3.unsqueeze(0), k4.unsqueeze(0)], dim=0)
        b_sum = torch.cat([self.baseline_conv.bias.unsqueeze(0), self.bias], dim=0)


        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = k_sum.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att[0], weight).view(bs * self.out_channels, self.in_channels,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k
        #cin_att = softmax_att[1].unsqueeze(1)
        cin_att = softmax_att[1].repeat(1, self.out_channels, 1, 1, 1).view(bs * self.out_channels, self.in_channels, 1, 1)
        aggregate_weight = cin_att * aggregate_weight

        #ks_att = softmax_att[2].unsqueeze(1)
        ks_att = softmax_att[2].repeat(1, self.out_channels, 1, 1, 1).view(bs * self.out_channels, 1, self.kernel_size, self.kernel_size)
        aggregate_weight = ks_att * aggregate_weight

        aggregate_weight = softmax_att[-1].view(bs * self.out_channels, 1, 1, 1) * aggregate_weight

        bias = b_sum.view(self.K, -1)  # K,out_p
        aggregate_bias = torch.mm(softmax_att[0], bias).view(-1)  # bs,out_p
        y = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=bs)
        y = y.view(bs, self.out_channels, h, w)

        return y #, y_225, y_270, y_315]

    def _rotate_kernel(self, original_kernel, angle):  # only 3*3 kernel
        n = angle // 45
        orig_tran = original_kernel.permute(2, 3, 0, 1)
        new_kernel = torch.zeros_like(orig_tran)
        if n == 1 :  # 45 degree
            new_kernel[0][1:][:][:] = orig_tran[0][:2][:][:]
            new_kernel[1][2][:][:] = orig_tran[0][2][:][:]
            new_kernel[2][2][:][:] = orig_tran[1][2][:][:]
            new_kernel[2][:2][:][:] = orig_tran[2][1:][:][:]
            new_kernel[0][0][:][:] = orig_tran[1][0][:][:]
            new_kernel[1][0][:][:] = orig_tran[2][0][:][:]
            new_kernel[1][1][:][:] = orig_tran[1][1][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 2:  # 90 degree
            l = len(orig_tran)
            for i in range(l):
                for j in range(l):
                    new_kernel[j][l - 1 - i][:][:] = orig_tran[i][j][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)


        if n == 3:  #  135 degree
            new_kernel[1][2][:][:] = orig_tran[0][0][:][:]
            new_kernel[2][2][:][:] = orig_tran[0][1][:][:]
            new_kernel[2][0][:][:] = orig_tran[1][2][:][:]
            new_kernel[2][1][:][:] = orig_tran[0][2][:][:]
            new_kernel[0][0][:][:] = orig_tran[2][1][:][:]
            new_kernel[1][0][:][:] = orig_tran[2][2][:][:]
            new_kernel[0][1][:][:] = orig_tran[2][0][:][:]
            new_kernel[0][2][:][:] = orig_tran[1][0][:][:]
            new_kernel[1][1][:][:] = orig_tran[1][1][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 4:  # 180 degree
            l = len(orig_tran)
            for i in range(l):
                for j in range(l):
                    new_kernel[i][j][:][:] = orig_tran[l - 1 - i][l - 1 - j][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 5:
            new_kernel[2][1][:][:] = orig_tran[0][0][:][:]
            new_kernel[2][0][:][:] = orig_tran[0][1][:][:]
            new_kernel[1][0][:][:] = orig_tran[0][2][:][:]
            new_kernel[0][0][:][:] = orig_tran[1][2][:][:]
            new_kernel[0][1][:][:] = orig_tran[2][2][:][:]
            new_kernel[0][2][:][:] = orig_tran[2][1][:][:]
            new_kernel[1][2][:][:] = orig_tran[2][0][:][:]
            new_kernel[2][2][:][:] = orig_tran[1][0][:][:]
            new_kernel[1][1][:][:] = orig_tran[1][1][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 6:  # 270 degree
            l = len(orig_tran)
            for i in range(l):
                for j in range(l):
                    new_kernel[l - 1 - j][i][:][:] = orig_tran[i][j][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 7:  # 315 degree
            new_kernel[1][0][:][:] = orig_tran[0][0][:][:]
            new_kernel[0][0][:][:] = orig_tran[0][1][:][:]
            new_kernel[0][1][:][:] = orig_tran[0][2][:][:]
            new_kernel[0][2][:][:] = orig_tran[1][2][:][:]
            new_kernel[1][2][:][:] = orig_tran[2][2][:][:]
            new_kernel[2][2][:][:] = orig_tran[2][1][:][:]
            new_kernel[2][1][:][:] = orig_tran[2][0][:][:]
            new_kernel[2][0][:][:] = orig_tran[1][0][:][:]
            new_kernel[1][1][:][:] = orig_tran[1][1][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        return new_kernel


class SKConv(nn.Module):
    def __init__(self, features, M, r, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.features = features
        self.bn_relus = nn.ModuleList()
        for i in range(M-1):  # 45, 90, 135, 180
            self.bn_relus.append(nn.Sequential(
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M-1):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        self.branch = M-1

    def forward(self, X):
        feas = []
        for i in range(self.branch):
            feas.append(self.bn_relus[i](X[i+1]).unsqueeze_(dim=1))
        feas = torch.cat(feas, dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v += X[0]  # residual structure
        return fea_v


@NECKS.register_module()
class FPN_AAR_DYN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 angle_nums = 5):
        super(FPN_AAR_DYN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.angle_nums = angle_nums

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_rotate_convs = nn.ModuleList()  # rotate_convolution , M angle branch
        self.angle_downsample_convs = nn.ModuleList()
        #self.fusions = nn.ModuleList()  # add fusion module SK-Net

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            fpn_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 45, 90, 135, 180]  # 225, 270, 315]
            )
            #fusion = SKConv(256, 5, 2, 32)

            if i > 1:  # P3, P4 downsample
                for kk in range(self.angle_nums):
                    angle_downsample_conv = ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.angle_downsample_convs.append(angle_downsample_conv)  # 0 45 90 135 180

            self.lateral_convs.append(l_conv)
            self.fpn_rotate_convs.append(fpn_rotate_conv)
            #self.fusions.append(fusion)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels

                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_rotate_convs.append(extra_fpn_conv)


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        # convolution: local
        outs = [
            self.fpn_rotate_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]  # P3-P4-P5


        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_rotate_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_rotate_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_rotate_convs[i](outs[-1]))
        return tuple(outs)