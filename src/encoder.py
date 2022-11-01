from typing import List

import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Conv1d :
        nn.init.xavier_uniform_(m.weight, gain=1.0)

    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        return x

class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()

        wrap = lambda v: [v] if type(v) is int else v
        kernel_size = wrap(kernel_size)
        dilation = wrap(dilation)
        stride = wrap(stride)

        new_kernel_size = max(kernel_size[0], 1)
        kernel_size = [new_kernel_size + 1 if new_kernel_size % 2 == 0 else new_kernel_size]

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.separable = separable

        infilters_loop = feat_in
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            conv.extend(
                self._get_conv_bn_layer(
                    infilters_loop, filters, kernel_size=kernel_size,
                    stride=stride, dilation=dilation, padding=padding_val,
                    groups=1, separable=separable)
            )
            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=None))
            
            infilters_loop = filters

        conv.extend(
            self._get_conv_bn_layer(
                infilters_loop, filters, kernel_size=kernel_size, stride=stride,
                dilation=dilation, padding=padding_val, groups=1, separable=separable)
        )
        self.conv = conv

        res_panes = []
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            res_panes = [feat_in]
            self.dense_residual = False
            for ip in res_panes:
                res_list.append(nn.ModuleList(
                    self._get_conv_bn_layer(ip, filters, kernel_size=1, stride=[1])
                ))

            self.res = res_list
        else:
            self.res = None

        self.out = nn.Sequential(*self._get_act_dropout_layer(
            drop_prob=dropout, activation=None))

    

    def _get_conv(self, in_channels, out_channels, kernel_size=11, stride=1,
                  dilation=1, padding=0, bias=False, groups=1):

        kw = {'in_channels': in_channels, 'out_channels': out_channels,
              'kernel_size': kernel_size, 'stride': stride, 'dilation': dilation,
              'padding': padding, 'bias': bias, 'groups': groups}

        return nn.Conv1d(**kw)

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False,
                           groups=1, separable=False):

        if separable:
            layers = [
                self._get_conv(in_channels, in_channels, kernel_size,
                               stride=stride, dilation=dilation, padding=padding,
                               bias=bias, groups=in_channels),
                self._get_conv(in_channels, out_channels, kernel_size=1,
                               stride=1, dilation=1, padding=0, bias=bias,
                               groups=groups),
            ]
        else:
            layers = [
                self._get_conv(in_channels, out_channels, kernel_size,
                               stride=stride, dilation=dilation,
                               padding=padding, bias=bias, groups=groups)
            ]

        layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers


    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        # compute forward convolutions
        out = xs[-1]
        for i, l in enumerate(self.conv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for res_layer in layer:
                    res_out = res_layer(res_out)

                out = out + res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        return out


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features.unsqueeze(0))
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len