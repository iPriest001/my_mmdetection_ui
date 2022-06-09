import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16, BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS


class GroupAttention(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1, init_cfg=None):
        """
        ws 1 for stand attention
        """
        super(GroupAttention, self).__init__(init_cfg)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    @auto_fp16()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads,
                                            C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, init_cfg=None):
        super().__init__(init_cfg)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    @auto_fp16()
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  #conv, maybe it can be replaced by pooling
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Cross_Attention(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, init_cfg=None):
        super().__init__(init_cfg)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    @auto_fp16()
    def forward(self, x, H, W, y, H1, W1):
        B, N, C = x.shape
        B1, N1, C1 = y.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B1, C1, H1, W1)
            y_ = self.sr(y_).reshape(B1, C1, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        else:
            kv = self.kv(y).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class self_attn(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop_path=0., attn_drop=0., proj_drop=0., ws=1, sr_ratio=1.0, init_cfg=None):
        super(self_attn, self).__init__(init_cfg)

        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.ws = ws
        self.sr_ratio = sr_ratio

        self.group_attn = GroupAttention(dim=self.dim, num_heads=self.num_heads, ws = self.ws)
        self.global_attn = Attention(dim=self.dim, num_heads=self.num_heads, sr_ratio=self.sr_ratio)  # self-attention
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @auto_fp16()
    def forward(self, x):
        B, C, H, W = x.size()
        x1 = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        x1 = x1 + self.drop_path(self.group_attn(self.layernorm1(x1), H, W))
        x1 = x1 + self.drop_path(self.global_attn(self.layernorm2(x1), H, W))
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # (B,C,H,W)

        return x1


class high2low_attn(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1, sr_ratio=1.0, init_cfg=None):
        super(high2low_attn, self).__init__(init_cfg)

        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.ws = ws
        self.sr_ratio = sr_ratio

        # attention operation
        self.group_attn = GroupAttention(dim=self.dim, num_heads=self.num_heads, ws=self.ws)
        self.global_attn = Cross_Attention(dim=self.dim, num_heads=self.num_heads, sr_ratio=self.sr_ratio)  # cross_attention
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.layernorm3 = nn.LayerNorm(dim)


    @auto_fp16()
    def forward(self, x_low, x_high):
        B, C, H, W = x_low.size()
        B1, C1, H1, W1 = x_high.size()
        x_low1 = x_low.reshape(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        x_high1 = x_high.reshape(B1, C1, -1).permute(0, 2, 1).contiguous()
        x_low1 = x_low1 + self.group_attn(self.layernorm1(x_low1), H, W)
        x_low1 = x_low1 + self.global_attn(self.layernorm2(x_low1), H, W, self.layernorm3(x_high1), H1, W1)
        x_low1 = x_low1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # (B,C,H,W)
        #x_high1 = x_high1.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()  # (B,C,H,W)

        return x_low1


class low2high_attn(BaseModule):
    def __init__(self, channels_high, channels_low, ratio, init_cfg=None):
        super(low2high_attn, self).__init__(init_cfg)
        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduction = nn.BatchNorm2d(channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.coorattention = cross_scale_CoordAtt(channels_low, channels_low, ratio)

    def forward(self, x_low, x_high):
        x_att = self.coorattention(x_low, x_high)
        out = self.relu(self.bn_reduction(self.conv1x1(x_high + x_att)))
        return out


# coordAttention !!!
class h_sigmoid(BaseModule):
    def __init__(self, inplace=True, init_cfg=None):
        super(h_sigmoid, self).__init__(init_cfg)
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(BaseModule):
    def __init__(self, inplace=True, init_cfg=None):
        super(h_swish, self).__init__(init_cfg)
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class cross_scale_CoordAtt(BaseModule):
    def __init__(self, inp, oup, ratio, reduction=32, init_cfg=None):
        super(cross_scale_CoordAtt, self).__init__(init_cfg)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.ratio = ratio

        mip = max(8, inp // reduction)
        if self.ratio == 2:
            self.conv1 = nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1)  # change k=1,s=1
        elif self.ratio == 4:
            self.conv1 = nn.Sequential(nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(inp, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.Conv2d(mip, mip, kernel_size=3, stride=2, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x_low, x_high):
        identity = x_high
        #n, c, h, w = x_low.size()
        n1, c1, h1, w1 = x_high.size()

        x_h = self.pool_h(x_low)
        x_w = self.pool_w(x_low).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h1, w1], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


# adaptive scale feature fusion
class ASFF(BaseModule):
    def __init__(self, rfb=False, vis=False, init_cfg=None):
        super(ASFF, self).__init__(init_cfg)
        self.dim = 256
        self.inter_dim = self.dim

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis
        self.expand = add_conv(self.inter_dim, 256, 3, 1)


    def forward(self, x_level_0, x_level_1, x_level_2):
        level_0_weight_v = self.weight_level_0(x_level_0)
        level_1_weight_v = self.weight_level_1(x_level_1)
        level_2_weight_v = self.weight_level_2(x_level_2)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v) ,1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = x_level_0 * levels_weight[:,0:1,:,:]+\
                            x_level_1 * levels_weight[:,1:2,:,:]+\
                            x_level_2 * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class AdaptiveAngleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, angle_list=[0]):
        super(AdaptiveAngleConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.angle_list = angle_list
        self.branches = len(angle_list)
        self.baseline_conv = nn.Conv2d(in_channel, out_channel, (kernel_size, kernel_size), (stride, stride), (padding, padding))

    def forward(self, x):
        y = self.baseline_conv(x)
        y_45 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[1]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        y_90 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[1]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        y_135 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[3]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        y_180 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[2]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        # y_225 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[5]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        # y_270 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[3]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        # y_315 = F.conv2d(x, self._rotate_kernel(self.baseline_conv.weight, self.angle_list[7]), bias=self.baseline_conv.bias, stride=self.stride, padding=self.padding)
        return [y, y_45, y_90, y_135, y_180]  # , y_225, y_270, y_315]
        # return [y, y_90, y_180, y_270]

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

        if n == 4:  # 180 degree
            l = len(orig_tran)
            for i in range(l):
                for j in range(l):
                    new_kernel[i][j][:][:] = orig_tran[l - 1 - i][l - 1 - j][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        if n == 6:  # 270 degree
            l = len(orig_tran)
            for i in range(l):
                for j in range(l):
                    new_kernel[l - 1 - j][i][:][:] = orig_tran[i][j][:][:]
            new_kernel = new_kernel.permute(2, 3, 0, 1)

        return new_kernel


class SKConv(nn.Module):
    def __init__(self, features, M, r, L=32):
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
class CSA_FPN_AAR2(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=True,  # use P6, P7
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 num_outs=5,  # in = out
                 with_norm=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_method='bilinear',
                 init_cfg=None,
                 angle_nums=5):
        super(CSA_FPN_AAR2, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.feature_dim = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_extra_convs = relu_before_extra_convs
        assert upsample_method in ['nearest', 'bilinear']
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

        if with_norm:
            self.fpna_p5_1x1 = nn.Sequential(
                *[nn.Conv2d(in_channels[3], out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
            self.fpna_p4_1x1 = nn.Sequential(
                *[nn.Conv2d(in_channels[2], out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
            self.fpna_p3_1x1 = nn.Sequential(
                *[nn.Conv2d(in_channels[1], out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
            # self.fpna_p2_1x1 = nn.Sequential(*[nn.Conv2d(in_channels[0], out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])

        else:
            self.fpna_p5_1x1 = nn.Conv2d(in_channels[3], out_channels, 1)
            self.fpna_p4_1x1 = nn.Conv2d(in_channels[2], out_channels, 1)
            self.fpna_p3_1x1 = nn.Conv2d(in_channels[1], out_channels, 1)

        # add attention
        # self_attention
        self.self_p3 = self_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=8)
        self.self_p4 = self_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=4)
        self.self_p5 = self_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=2)
        # high_to_low attention
        self.h2l_p4_p3 = high2low_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=4)
        self.h2l_p5_p3 = high2low_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=2)
        self.h2l_p5_p4 = high2low_attn(dim=out_channels, num_heads=8, ws=7, sr_ratio=2)
        # low_to_high attention
        self.l2h_p3_p4 = low2high_attn(out_channels, out_channels, 2)
        self.l2h_p3_p5 = low2high_attn(out_channels, out_channels, 4)
        self.l2h_p4_p5 = low2high_attn(out_channels, out_channels, 2)

        # adaptive feature fusion
        self.p3_fusion = ASFF(rfb=False, vis=False)
        self.p4_fusion = ASFF(rfb=False, vis=False)
        self.p5_fusion = ASFF(rfb=False, vis=False)

        # !!!!!!!!!!!! FPN_AAR  !!!!!!!!!!!!!!!
        self.fpn_rotate_convs = nn.ModuleList()  # rotate_convolution , M angle branch
        self.angle_downsample_convs = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.aar_fusions = nn.ModuleList()  # add fusion module SK-Net
        for i in range(self.start_level, self.backbone_end_level):
            fpn_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 90, 180, 270]
            )
            aar_fusion = SKConv(256, 5, 2, 32)

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

                    conv_1x1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
                    self.convs_1x1.append(conv_1x1)

            self.fpn_rotate_convs.append(fpn_rotate_conv)
            self.aar_fusions.append(aar_fusion)

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
        assert len(inputs) == len(self.in_channels)
        p5 = self.fpna_p5_1x1(inputs[self.end_level])
        p4 = self.fpna_p4_1x1(inputs[self.start_level + 1])
        p3 = self.fpna_p3_1x1(inputs[self.start_level])

        fpna_p3_out = self.p3_fusion(self.self_p3(p3), self.h2l_p4_p3(p3, p4), self.h2l_p5_p3(p3, p5))
        fpna_p4_out = self.p4_fusion(self.self_p4(p4), self.h2l_p5_p4(p4, p5), self.l2h_p3_p4(p3, p4))
        fpna_p5_out = self.p5_fusion(self.self_p5(p5), self.l2h_p3_p5(p3, p5), self.l2h_p4_p5(p4, p5))

        fpna_outs= [fpna_p3_out, fpna_p4_out, fpna_p5_out]

        # convolution: local
        # !!!!!!!!!! FPN-AAR !!!!!!!!!!!!!!!!!!
        used_backbone_levels = self.num_ins - self.start_level
        conv_outs = [
            self.fpn_rotate_convs[i](fpna_outs[i]) for i in range(used_backbone_levels)
        ]  # P3-P4-P5

        for i in range(1, used_backbone_levels):
            for j in range(self.angle_nums):
                downsample_num = 5 + j if i>1 else j
                temp = self.angle_downsample_convs[downsample_num](conv_outs[i-1][j])
                conv_outs[i][j] = self.convs_1x1[downsample_num](torch.cat([conv_outs[i][j], temp], dim=1))

        # local-global fusion
        outs = []
        for i in range(used_backbone_levels):
            outs.append(self.aar_fusions[i](conv_outs[i]) + fpna_outs[i]) # redisual structure


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
                elif self.add_extra_convs == 'after_CSA_FPN':  #  after CSA-FPN
                    extra_source = fpna_outs[-1]
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

