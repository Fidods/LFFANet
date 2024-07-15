import math

import torch
from torch import nn
class Activation(nn.Module):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.Hardswish()
        else:
            raise NotImplementedError

    def forward(self, x):
        """ forward """
        return self.act(x)
class ConvUnit(nn.Module):
    """
    ConvUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (Union[int, tuple[int]]): Input kernel size.
        stride (int): Stride size.
        padding (Union[int, tuple[int]]): Padding number.
        num_groups (int): Output num group.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvUnit(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=num_groups,
                              bias=False,
                              padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def forward(self, x):
        """ forward of conv unit """
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GhostModule(nn.Module):
    """
    GhostModule warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostModule(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size//2, num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)

    def forward(self, x):
        """ ghost module forward """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
class maxpool_LISDNet_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(maxpool_LISDNet_Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.BN = nn.BatchNorm2d(out_size)


        self.shortcut = nn.Sequential()
        self.down=nn.MaxPool2d(kernel_size=2)
        self.Ghost=GhostModule(in_size,out_size)
        self.save=nn.Conv2d(out_size,out_size,1,1,0)
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        self.gate_conv=nn.Conv2d(out_size*2,out_size,1,1,0)

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.stride==2:
            out=self.down(out)

        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        if self.stride==2:
            x=self.down(x)
        out2=self.Ghost(x)
        out2=self.save(out2)

        final=self.nolinear1(self.gate_conv(torch.cat([out,out2],1)))
        return final
class LISDNet_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(LISDNet_Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.BN = nn.BatchNorm2d(out_size)


        self.shortcut = nn.Sequential()
        self.down=nn.MaxPool2d(kernel_size=2)
        self.Ghost=GhostModule(in_size,out_size)
        self.save=nn.Conv2d(out_size,out_size,1,1,0)
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        self.gate_conv=nn.Conv2d(out_size*2,out_size,1,1,0)

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        if self.stride==2:
            x=self.down(x)
        out2=self.Ghost(x)
        out2=self.save(out2)

        final=self.nolinear1(self.gate_conv(torch.cat([out,out2],1)))

class convBlock(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(convBlock, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)


        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.shortcut(x) if self.stride==1 else out

        return out
class Su_LISDNet_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Su_LISDNet_Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size//2, expand_size//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size//2)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size//2, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)



        self.shortcut = nn.Sequential()
        self.down=nn.MaxPool2d(kernel_size=2)
        self.Ghost=GhostModule(in_size//2,out_size)
        self.save=nn.Conv2d(out_size,out_size,1,1,0)

        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size//2, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        self.gate_conv=nn.Conv2d(out_size*2,out_size,kernel_size=1,stride=1,padding=0)

    def channel_shuffle(self,x, groups):
        # 获取输入特征图的shape=[b,c,h,w]
        batch_size, num_channels, height, width = x.size()
        # 均分通道，获得每个组对应的通道数
        channels_per_group = num_channels // groups
        # 特征图shape调整 [b,c,h,w]==>[b,g,c_g,h,w]
        x = x.view(batch_size, groups, channels_per_group, height, width)
        # 维度调整 [b,g,c_g,h,w]==>[b,c_g,g,h,w]；将调整后的tensor以连续值的形式保存在内存中
        x = torch.transpose(x, 1, 2).contiguous()
        # 将调整后的通道拼接回去 [b,c_g,g,h,w]==>[b,c,h,w]
        x = x.view(batch_size, -1, height, width)
        # 完成通道重排
        return x
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = self.nolinear1(self.bn1(self.conv1(x1)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        if self.stride==2:
            x2=self.down(x2)
        out2=self.Ghost(x2)
        out2=self.save(out2)
        out3=self.channel_shuffle(torch.cat([out,out2],1),2)
        final=self.nolinear1(self.gate_conv(out3))
        return final
class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.ReLU1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.ReLU1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ReLU = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):

        out = self.ca(x) * x
        out = self.sa(out) * out
        out = self.ReLU(out)
        return out
class stem(nn.Module):
    def __init__(self,inch,outch):
        super(stem,self).__init__()
        self.inch=inch
        self.outch=outch
        self.conv_1=nn.Conv2d(inch,8,3,1,1)
        self.bn_1=nn.BatchNorm2d(8)
        self.relu=nn.ReLU(inplace=True)
        self.conv_2=nn.Conv2d(8,8,3,1,1)
        self.bn_2 = nn.BatchNorm2d(8)
        self.res=nn.Conv2d(3,8,1,1,0)

    def forward(self,x):
        res=self.res(x)
        x=self.relu(self.bn_1(self.conv_1(x)))
        x=self.relu(self.bn_2(self.conv_2(x))+res)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class DY_cefenzhi_ghostbackbone(nn.Module):
    def __init__(self,inch):
        super(DY_cefenzhi_ghostbackbone,self).__init__()
        self.inch=inch
        self.conv=stem(inch,8)
        self.layer_1=nn.Sequential(DY_MobileNetBlock(3,8,16,16,nn.ReLU(inplace=True), CBAM_block(16,16),2),
                                   DY_MobileNetBlock(3,16,32,16,nn.ReLU(inplace=True),CBAM_block(16,16),1))
        self.layer_2=nn.Sequential(DY_MobileNetBlock(3,16,32,32,nn.ReLU(inplace=True),CBAM_block(32,32),2),
                                   DY_MobileNetBlock(3,32,64,32,nn.ReLU(inplace=True), CBAM_block(32,32),1))
        self.layer_3=nn.Sequential(DY_MobileNetBlock(3,32,64,64,hswish(), CBAM_block(64,64),2),
                                   DY_MobileNetBlock(3,64,128,64,hswish(), CBAM_block(64,64),1))
        self.layer_4=nn.Sequential(DY_MobileNetBlock(3,64,128,128,hswish(), CBAM_block(128,128),2),
                                   DY_MobileNetBlock(3,128,256,128,hswish(), CBAM_block(128,128),1))

    def forward(self,x):

        out1=self.conv(x)
        out2=self.layer_1(out1)
        out3=self.layer_2(out2)
        out4=self.layer_3(out3)
        out5=self.layer_4(out4)
        return [out1,out2,out3,out4,out5]
class LISDNet_backbone(nn.Module):
    def __init__(self,inch):
        super(LISDNet_backbone,self).__init__()
        self.inch=inch
        self.conv=stem(inch,8)
        self.layer_1=nn.Sequential(LISDNet_Block(3,8,16,16,nn.ReLU(inplace=True), CBAM_block(16,16),2),
                                   LISDNet_Block(3,16,32,16,nn.ReLU(inplace=True),CBAM_block(16,16),1))
        self.layer_2=nn.Sequential(LISDNet_Block(3,16,32,32,nn.ReLU(inplace=True),CBAM_block(32,32),2),
                                   LISDNet_Block(3,32,64,32,nn.ReLU(inplace=True), CBAM_block(32,32),1))
        self.layer_3=nn.Sequential(LISDNet_Block(3,32,64,64,hswish(), CBAM_block(64,64),2),
                                   LISDNet_Block(3,64,128,64,hswish(), CBAM_block(64,64),1))
        self.layer_4=nn.Sequential(LISDNet_Block(3,64,128,128,hswish(), CBAM_block(128,128),2),
                                   LISDNet_Block(3,128,256,128,hswish(), CBAM_block(128,128),1))

    def forward(self,x):

        out1=self.conv(x)
        out2=self.layer_1(out1)
        out3=self.layer_2(out2)
        out4=self.layer_3(out3)
        out5=self.layer_4(out4)
        return [out1,out2,out3,out4,out5]
class conv_blockbanckbone(nn.Module):
    def __init__(self,inch):
        super(conv_blockbanckbone,self).__init__()
        self.inch=inch
        self.conv=stem(inch,8)
        self.layer_1=nn.Sequential(convBlock(3,8,16,16,nn.ReLU(inplace=True), None,2),
                                   convBlock(3,16,32,16,nn.ReLU(inplace=True),None,1))
        self.layer_2=nn.Sequential(convBlock(3,16,32,32,nn.ReLU(inplace=True),None,2),
                                   convBlock(3,32,64,32,nn.ReLU(inplace=True), None,1))
        self.layer_3=nn.Sequential(convBlock(3,32,64,64,hswish(), None,2),
                                   convBlock(3,64,128,64,hswish(), None,1))
        self.layer_4=nn.Sequential(convBlock(3,64,128,128,hswish(), None,2),
                                   convBlock(3,128,256,128,hswish(), None,1))

    def forward(self,x):

        out1=self.conv(x)
        out2=self.layer_1(out1)
        out3=self.layer_2(out2)
        out4=self.layer_3(out3)
        out5=self.layer_4(out4)
        return [out1,out2,out3,out4,out5]
if __name__ == '__main__':
    x=torch.rand(4,3,256,256)
    net=LISDNet_Block(3)
    x=x.cuda()
    net=net.cuda()
    out=net(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)



    #print(out)
