import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS


class AdaptiveAngleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, angle_list=[0]):
        super(AdaptiveAngleConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.angle_list = angle_list
        self.branches = len(angle_list)
        self.baseline_conv = nn.Conv2d(in_channel, out_channel, (kernel_size, kernel_size), (stride, stride), (padding, padding), dilation=self.dilation)

    def forward(self, x):
        y = self.baseline_conv(x)
        y_45 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[1]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        y_90 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[2]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        y_135 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[3]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        y_180 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[4]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # y_225 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[5]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # y_270 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[6]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # y_315 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[7]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return [y, y_45, y_90, y_135, y_180]  #, y_225, y_270, y_315]

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
class FPN_AAR3(nn.Module):
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
        super(FPN_AAR3, self).__init__()
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
        self.angle_downsample_convs = nn.ModuleList()
        self.fusions = nn.ModuleList()  # add fusion module SK-Net

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fusion = SKConv(256, 5, 2, 32)

            if i>1:  # P3,P4 downsample
                for angle in range(self.angle_nums):
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
            self.fusions.append(fusion)

        # P3, dilation=1, 3, 5; rotate_convolution , M angle branch
        self.p3_dilation_rotate_convs = nn.ModuleList()
        self.p3_concat_1x1_convs = nn.ModuleList()
        for d in range(1, 6, 2):
            p3_dilation_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                padding=d,
                dilation=d,
                angle_list=[0, 45, 90, 135, 180]  # 225, 270, 315]
            )
            self.p3_dilation_rotate_convs.append(p3_dilation_rotate_conv)

        for angle in range(self.angle_nums):
            p3_concat_1x1_conv = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1, padding=0)
            self.p3_concat_1x1_convs.append(p3_concat_1x1_conv)

        # p4, dilation=1, 3; rotate_convolution , M angle branch
        self.p4_dilation_rotate_convs = nn.ModuleList()
        self.p4_concat_1x1_convs = nn.ModuleList()
        for d in range(1, 4, 2):
            p4_dilation_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                padding=d,
                dilation=d,
                angle_list=[0, 45, 90, 135, 180]  # 225, 270, 315]
            )
            self.p4_dilation_rotate_convs.append(p4_dilation_rotate_conv)

        for angle in range(self.angle_nums):
            p4_concat_1x1_conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
            self.p4_concat_1x1_convs.append(p4_concat_1x1_conv)

        # p5
        self.p5_dilation_rotate_convs = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 45, 90, 135, 180]  # 225, 270, 315]
            )  # dilation=1

        # add extra conv layers (e.g., RetinaNet)
        # add extra conv layers (e.g., RetinaNet)
        if self.add_extra_convs == 'on_input':
            self.fpn_p6 = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=2, padding=1)
            self.fpn_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.fpn_p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.fpn_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)


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
        p3_dilation_conv_outs = [self.p3_dilation_rotate_convs[i](laterals[0]) for i in range(3)] # d=1,3,5
        p4_dilation_conv_outs = [self.p4_dilation_rotate_convs[i](laterals[1]) for i in range(2)] # d=1,3
        p5_outs = self.p5_dilation_rotate_convs(laterals[2])

        # merge p3 0/45/90/135/180 degree branches
        p3_outs = []
        for angle in range(self.angle_nums):
            temp = torch.cat([p3_dilation_conv_outs[0][angle], p3_dilation_conv_outs[1][angle], p3_dilation_conv_outs[2][angle]], dim=1)
            temp = self.p3_concat_1x1_convs[angle](temp)
            p3_outs.append(temp)

        # merge p4 0/45/90/135/180 degree branches
        p4_outs = []
        for angle in range(self.angle_nums):
            temp = torch.cat([p4_dilation_conv_outs[0][angle], p4_dilation_conv_outs[1][angle]], dim=1)
            temp = self.p4_concat_1x1_convs[angle](temp)
            p4_outs.append(temp)

        conv_outs = [p3_outs, p4_outs, p5_outs]

        for i in range(1, used_backbone_levels):
            for j in range(self.angle_nums):
                downsample_num = 5 + j if i > 1 else j
                conv_outs[i][j] += self.angle_downsample_convs[downsample_num](
                    conv_outs[i - 1][j])  # directly add (e.g. p3_45_downsample + p4_45)

        # local-global fusion
        outs = [
            self.fusions[i](conv_outs[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.add_extra_convs == 'on_input':
            fpn_p6_out = self.fpn_p6(inputs[-1])
            fpn_p7_out = self.fpn_p7(fpn_p6_out)
        else:
            fpn_p6_out = self.fpn_p6(outs[-1])
            fpn_p7_out = self.fpn_p7(fpn_p6_out)

        outs.extend([fpn_p6_out, fpn_p7_out])

        return tuple(outs)