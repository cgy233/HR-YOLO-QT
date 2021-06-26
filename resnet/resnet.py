import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'wide_resnet101_2'
]

model_urls = {
    'resnet18_lpf2':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf2-6e2ee76f.pth',
    'resnet18_lpf3':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf3-449351b9.pth',
    'resnet18_lpf4':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4-8c77af40.pth',
    'resnet18_lpf5':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf5-c1eed0a1.pth',
    'resnet34_lpf2':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf2-4707aed9.pth',
    'resnet34_lpf3':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf3-16aa6c48.pth',
    'resnet34_lpf4':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf4-55747267.pth',
    'resnet34_lpf5':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf5-85283561.pth',
    'resnet50_lpf2':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf2-f0f7589d.pth',
    'resnet50_lpf3':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf3-a4e868d2.pth',
    'resnet50_lpf4':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4-994b528f.pth',
    'resnet50_lpf5':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf5-9953c9ad.pth',
    'resnet101_lpf2':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf2-3d00941d.pth',
    'resnet101_lpf3':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf3-928f1444.pth',
    'resnet101_lpf4':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf4-f8a116ff.pth',
    'resnet101_lpf5':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf5-1f3745af.pth',
    'resnet18_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4_finetune-8cc58f59.pth',
    'resnet34_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf4_finetune-db622952.pth',
    'resnet50_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4_finetune-cad66808.pth',
    'resnet101_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf4_finetune-9280acb0.pth',
    'resnet152_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet152_lpf4_finetune-7f67d9ae.pth',
    'resnext50_32x4d_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnext50_32x4d_lpf4_finetune-9106e549.pth',
    'resnext101_32x8d_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnext101_32x8d_lpf4_finetune-8f13a25d.pth',
    'wide_resnet50_2_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/wide_resnet50_2_lpf4_finetune-02a183f7.pth',
    'wide_resnet101_2_lpf4_finetune':
    'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/wide_resnet101_2_lpf4_finetune-da4eae04.pth',
}


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BlurPool(nn.Module):
    def __init__(self,
                 channels,
                 pad_type='reflect',
                 filt_size=4,
                 stride=2,
                 pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([
                1.,
            ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp),
                            self.filt,
                            stride=self.stride,
                            groups=inp.shape[1])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if (stride == 1):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(
                BlurPool(planes, filt_size=filter_size, stride=stride),
                conv3x3(planes, planes),
            )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width, width, groups=groups, dilation=dilation
        )  # Conv(stride2)-Norm-Relu --> #Conv-Norm-Relu-BlurPool(stride2)
        self.bn2 = norm_layer(width)
        if (stride == 1):
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(
                BlurPool(width, filt_size=filter_size, stride=stride),
                conv1x1(width, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None,
                 filter_size=1,
                 pool_only=True,
                 replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        if (pool_only):
            self.conv1 = nn.Conv2d(3,
                                   self.inplanes,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=2, stride=1),
                BlurPool(
                    self.inplanes,
                    filt_size=filter_size,
                    stride=2,
                )
            ])
        else:
            self.conv1 = nn.Conv2d(3,
                                   self.inplanes,
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Sequential(*[
                BlurPool(
                    self.inplanes,
                    filt_size=filter_size,
                    stride=2,
                ),
                nn.MaxPool2d(kernel_size=2, stride=1),
                BlurPool(
                    self.inplanes,
                    filt_size=filter_size,
                    stride=2,
                )
            ])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       filter_size=filter_size)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       filter_size=filter_size)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       filter_size=filter_size)
        self.avgpool = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=7, groups=512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (m.in_channels != m.out_channels
                        or m.out_channels != m.groups or m.bias is not None):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    filter_size=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # since this is just a conv1x1 layer (no nonlinearity),
            # conv1x1->blurpool is the same as blurpool->conv1x1; the latter is cheaper
            downsample = [
                BlurPool(filt_size=filter_size,
                         stride=stride,
                         channels=self.inplanes),
            ] if (stride != 1) else []
            downsample += [
                conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)
            ]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  base_width=self.base_width,
                  dilation=previous_dilation,
                  norm_layer=norm_layer,
                  filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      filter_size=filter_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x.reshape(-1,21,2)


def resnet18(pretrained=False,
             filter_size=4,
             pool_only=True,
             _force_nonfinetuned=False,
             **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4 and not _force_nonfinetuned):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet18_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet18_lpf%i' % filter_size],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
    return model


def resnet34(pretrained=False,
             filter_size=4,
             pool_only=True,
             _force_nonfinetuned=False,
             **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
        _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4 and not _force_nonfinetuned):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet34_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet34_lpf%i' % filter_size],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
    return model


def resnet50(pretrained=False,
             filter_size=4,
             pool_only=True,
             _force_nonfinetuned=False,
             **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
        _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4 and not _force_nonfinetuned):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet50_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet50_lpf%i' % filter_size],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
    return model


def resnet101(pretrained=False,
              filter_size=4,
              pool_only=True,
              _force_nonfinetuned=False,
              **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
        _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4 and not _force_nonfinetuned):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet101_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet101_lpf%i' % filter_size],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
    return model


def resnet152(pretrained=False,
              filter_size=4,
              pool_only=True,
              _force_nonfinetuned=False,
              **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
        _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnet152_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def resnext50_32x4d(pretrained=False,
                    filter_size=4,
                    pool_only=True,
                    _force_nonfinetuned=False,
                    **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   groups=32,
                   width_per_group=4,
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4):
            model.load_state_dict(
                model_zoo.load_url(model_urls['resnext50_32x4d_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def resnext101_32x8d(pretrained=False,
                     filter_size=4,
                     pool_only=True,
                     _force_nonfinetuned=False,
                     **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   groups=32,
                   width_per_group=8,
                   filter_size=filter_size,
                   pool_only=pool_only,
                   **kwargs)
    if pretrained:
        if (filter_size == 4):
            model.load_state_dict(
                model_zoo.load_url(
                    model_urls['resnext101_32x8d_lpf4_finetune'],
                    map_location='cpu',
                    check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def wide_resnet50_2(pretrained=False,
                    filter_size=4,
                    _force_nonfinetuned=False,
                    **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   width_per_group=64 * 2,
                   filter_size=filter_size,
                   **kwargs)
    if pretrained:
        if (filter_size == 4):
            model.load_state_dict(
                model_zoo.load_url(model_urls['wide_resnet50_2_lpf4_finetune'],
                                   map_location='cpu',
                                   check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


def wide_resnet101_2(pretrained=False,
                     filter_size=4,
                     _force_nonfinetuned=False,
                     **kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   width_per_group=64 * 2,
                   filter_size=filter_size,
                   **kwargs)
    if pretrained:
        if (filter_size == 4):
            model.load_state_dict(
                model_zoo.load_url(
                    model_urls['wide_resnet101_2_lpf4_finetune'],
                    map_location='cpu',
                    check_hash=True)['state_dict'])
        else:
            raise ValueError('No pretrained model available')
    return model


if __name__ == "__main__":
    model = wide_resnet50_2(pretrained=False, filter_size=5)
    checkpoint = torch.load("models/wide_resnet50_2.pth")
    # state_dict = model.state_dict()
    # print(len(checkpoint))

    # new_state_dict = {}
    # for k, v in checkpoint.items():
    #     if checkpoint[k].shape == state_dict[k].shape:
    #         new_state_dict[k] = v
    # print(len(new_state_dict))
    # model.load_state_dict(new_state_dict, strict=False)
    # print(checkpoint.keys())
    # print(checkpoint['maxpool.1.filt'].shape)
    # print(checkpoint['maxpool.1.filt'][0])
    # torch.save(model.state_dict(), "models/wide_resnet50_2.pth")