import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer
import timm

from mmengine.model import BaseModule
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

from .decode_heads import PSPHead


class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level high-resolution feature map from encoder), it does not
            need upsample the high-level feature map and the upsample_cfg is
            None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None,
                 add_feature_map=False):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.add_feature_map = add_feature_map
        if not add_feature_map:
            self.conv_block = conv_block(
                in_channels=2 * skip_channels,
                out_channels=out_channels,
                num_convs=num_convs,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dcn=None,
                plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        if self.add_feature_map:
            out = skip + x
        else:
            out = torch.cat([skip, x], dim=1)
            out = self.conv_block(out)
        return out


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None,
                 dws=False):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    groups=in_channels if dws else 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


class MultimodalConcat(nn.Module):
    def __init__(self, n_channels, is_full=True, scale_aligned=False):
        super(MultimodalConcat, self).__init__()
        self.scale_aligned = scale_aligned
        if scale_aligned:
            self.s2_scale = nn.BatchNorm2d(n_channels)
            self.s1_scale = nn.BatchNorm2d(n_channels)
            self.srtm_scale = nn.BatchNorm2d(n_channels)
        ratio = 2
        if is_full:
            ratio += 1
        self.out = nn.Sequential(
            nn.Conv2d(n_channels * ratio, n_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )

    def forward(self, s2, s1, topo=None):
        if self.scale_aligned:
            s2 = self.s2_scale(s2)
            s1 = self.s1_scale(s1)
            topo = self.srtm_scale(topo)

        s2_s1_srtm = torch.cat((s2, s1, topo), 1)


        return self.out(s2_s1_srtm)


class ResEncoder(nn.Module):
    def __init__(self,
                 in_channels=10,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=2
                 ):
        super().__init__()

        self.initial_block = BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=64,
                    num_convs=2,
                    stride=1,
                    dilation=1,
                    with_cp=False,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    dcn=None,
                    plugins=None)

        self.encoder = timm.create_model(backbone_name, features_only=True, output_stride=16,
                                          out_indices=(0, 1, 2, 3), pretrained=False)
        original_first_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )

        if pretrained:
            pretrained_state_dict = timm.create_model(backbone_name, pretrained=True).state_dict()
            if 'conv1.weight' in pretrained_state_dict:
                del pretrained_state_dict['conv1.weight']
            if 'conv1.bias' in pretrained_state_dict:
                del pretrained_state_dict['conv1.bias']
            self.encoder.load_state_dict(pretrained_state_dict, strict=False)


    def forward(self, x):
        x = self.initial_block(x)
        features = self.encoder(x)
        return [x] + features

    def get_channels(self):
        return [64] + self.encoder.feature_info.channels()


class UNetResMultiEnc(BaseModule):
    def __init__(self,
                 s2_in_channels=4, s1_in_channels=4, topo_in_channels=2,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 backbone_name='swsl_resnet18',
                 kw_branch=True):
        super(UNetResMultiEnc, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages,'\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages,'\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.s2_encoder = ResEncoder(in_channels=s2_in_channels, backbone_name=backbone_name, pretrained=False)
        self.s1_encoder = ResEncoder(in_channels=s1_in_channels, backbone_name=backbone_name, pretrained=False)
        self.topo_encoder = ResEncoder(in_channels=topo_in_channels, backbone_name=backbone_name, pretrained=False)

        encoder_channels = self.s2_encoder.get_channels()

        self.ffm = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            if i != 0:
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=encoder_channels[i],
                        skip_channels=encoder_channels[i-1],
                        out_channels=encoder_channels[i-1],
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))
            self.ffm.append(MultimodalConcat(encoder_channels[i], scale_aligned=True))

        self.decode_head = PSPHead(in_channels=64,
                                in_index=4,
                                channels=16,
                                pool_scales=(1, 2, 3, 6),
                                dropout_ratio=0.1,
                                num_classes=2,
                                norm_cfg=norm_cfg,
                                align_corners=False)
        self.kw_branch = kw_branch
        if kw_branch:
            self.kw_decode_head = SimpleSegmentationHead(in_channels=64, num_classes=2)

    def forward(self, s2, s1, topo):
        self._check_input_divisible(s2)
        self._check_input_divisible(s1)
        self._check_input_divisible(topo)

        s2_enc_outs = self.s2_encoder(s2)
        s1_enc_outs = self.s1_encoder(s1)
        topo_enc_outs = self.topo_encoder(topo)
        s2, s1, topo = s2_enc_outs[-1], s1_enc_outs[-1], topo_enc_outs[-1]
        x = self.ffm[self.num_stages-1](s2, s1, topo)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            enc_outs = self.ffm[i](s2_enc_outs[i], s1_enc_outs[i], topo_enc_outs[i])
            x = self.decoder[i](enc_outs, x)
            dec_outs.append(x)

        output = self.decode_head(dec_outs)

        if not self.training:
            return output

        if self.kw_branch:
            kw_output = self.kw_decode_head(dec_outs[-1])
            return kw_output, output
        else:
            return output


    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
