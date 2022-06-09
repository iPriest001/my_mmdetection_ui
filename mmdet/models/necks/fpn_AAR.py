import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS


class AdaptiveAngleConv(nn.Module):  # deformable convolution version
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, angle_list=[0]):
        super(AdaptiveAngleConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # self.baseline_conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.baseline_conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=1, bias=bias)
        self.angle_list = angle_list
        self.branches = len(angle_list)


    def forward(self, x):
        b, c, h, w = x.shape
        N = self.kernel_size * self.kernel_size
        x_offset_0 = self._get_x_offset(x, N, h, w, self.angle_list[0])
        x_offset_45 = self._get_x_offset(x, N, h, w, self.angle_list[1])
        x_offset_90 = self._get_x_offset(x, N, h, w, self.angle_list[2])
        x_offset_135 = self._get_x_offset(x, N, h, w, self.angle_list[3])
        x_offset_180 = self._get_x_offset(x, N, h, w, self.angle_list[4])

        y = self.baseline_conv(x_offset_0)
        y_45 = F.conv2d(x_offset_45, self.baseline_conv.weight, bias=self.baseline_conv.bias, stride=self.baseline_conv.stride,
                        padding=self.baseline_conv.padding)
        y_90 = F.conv2d(x_offset_90, self.baseline_conv.weight, bias=self.baseline_conv.bias, stride=self.baseline_conv.stride,
                        padding=self.baseline_conv.padding)
        y_135 = F.conv2d(x_offset_135, self.baseline_conv.weight, bias=self.baseline_conv.bias, stride=self.baseline_conv.stride,
                         padding=self.baseline_conv.padding)
        y_180 = F.conv2d(x_offset_180, self.baseline_conv.weight, bias=self.baseline_conv.bias, stride=self.baseline_conv.stride,
                         padding=self.baseline_conv.padding)

        return [y, y_45, y_90, y_135, y_180]

    def _get_x_offset(self, x, N, h, w, angle):  # result before common convolution
        dtype = x.data.type()
        offset = self._get_p_offset(N, h, w, angle, dtype)
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, N=N, h=h, w=w, dtype=dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb

        x_offset += g_lb.unsqueeze(dim=1) * x_q_lb
        x_offset += g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)

        return x_offset

    def _get_p_offset(self, N, h, w, angle, dtype):
        n = angle // 45
        sqrt_2 = 2**0.5
        if n == 0:  # 0 degree
            offset_x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            offset_y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if n == 1:  # 45 degree
            offset_x = [1 - sqrt_2, 1 - sqrt_2 * 0.5, 1, -sqrt_2 * 0.5, 0, sqrt_2 * 0.5, -1, sqrt_2 * 0.5 - 1,
                        sqrt_2 - 1]
            offset_y = [1, sqrt_2 * 0.5, sqrt_2 - 1, 1 - sqrt_2 * 0.5, 0, sqrt_2 * 0.5 - 1, 1 - sqrt_2, -sqrt_2 * 0.5,
                        -1]
        if n == 2:  # 90 degree
            offset_x = [0, 1, 2, -1, 0, 1, -2, -1, 0]
            offset_y = [2, 1, 0, 1, 0, -1, 0, -1, -2]
        if n == 3:  # 135 degree
            offset_x = [1, 1 + sqrt_2 * 0.5, 1 + sqrt_2, -sqrt_2 * 0.5, 0, sqrt_2 * 0.5, -1 - sqrt_2, -1 - sqrt_2 * 0.5,
                        -1]
            offset_y = [1 + sqrt_2, sqrt_2 * 0.5, -1, 1 + sqrt_2 * 0.5, 0, -1 - sqrt_2 * 0.5, 1, -sqrt_2 * 0.5,
                        1 + sqrt_2]
        if n == 4:  # 180 degree
            offset_x = [2, 2, 2, 0, 0, 0, -2, -2, -2]
            offset_y = [2, 0, -2, 2, 0, -2, 2, 0, -2]
        offset_x = torch.tensor(offset_x)
        offset_y = torch.tensor(offset_y)
        offset = torch.cat([torch.flatten(offset_x), torch.flatten(offset_y)], 0)
        offset = offset.view(1, 2 * N, 1, 1).type(dtype)  # offset.view(1, 2 * N, 1, 1)

        offset = offset.repeat(2, 1, 1, 1)

        return offset

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, N, h, w, dtype):
        # N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


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
class FPN_AAR(nn.Module):
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
                 upsample_cfg=dict(mode='nearest')):
        super(FPN_AAR, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

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

            fpn_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 45, 90, 135, 180]
            )
            fusion = SKConv(256, 5, 2, 32)

            self.lateral_convs.append(l_conv)
            self.fpn_rotate_convs.append(fpn_rotate_conv)
            self.fusions.append(fusion)

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
        conv_outs = [
            self.fpn_rotate_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # local-global fusion
        outs = [
            self.fusions[i](conv_outs[i]) for i in range(used_backbone_levels)
        ]

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