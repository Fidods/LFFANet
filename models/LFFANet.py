import math

from torch.nn import init

from models.backbone import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv2d') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
def my_init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
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

class seCBAM_block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(seCBAM_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)  # in relu inplace = true
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1,padding=1,groups=out_channels//8)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*2, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(out_channels*2))


        self.ca = ChannelAttention(out_channels*2)
        self.sa = SpatialAttention()
        # self.dropout = nn.Dropout(0.3) # Dropout

    def forward(self, x):

        residual = self.shortcut(x)
        #print(residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        #print(out.shape)
        out += residual
        out=self.bn3(self.conv3(out))
        out = self.ReLU(out)

        return out
class seCBAM_block_noattn(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(seCBAM_block_noattn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)  # in relu inplace = true
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1,padding=1,groups=out_channels//8)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*2, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(out_channels*2))


        # self.ca = ChannelAttention(out_channels*2)
        # self.sa = SpatialAttention()
        # self.dropout = nn.Dropout(0.3) # Dropout

    def forward(self, x):

        residual = self.shortcut(x)
        #print(residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        #print(out.shape)
        out += residual
        out=self.bn3(self.conv3(out))
        out = self.ReLU(out)

        return out


class CONV(nn.Module):
    def __init__(self,inch,ouch,k,s,p):
        super(CONV,self).__init__()
        self.conv=nn.Conv2d(inch,ouch,k,s,p)
        self.BN=nn.BatchNorm2d(ouch)
        self.relu=nn.ReLU(inplace=True)
        #initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                my_init_weights(m, init_type='xavier')
            elif isinstance(m, nn.BatchNorm2d):
                my_init_weights(m, init_type='xavier')
    def forward(self,x):
        return self.relu(self.BN(self.conv(x)))






class LFFANet(nn.Module):

    def __init__(self, in_channels=3):
        super(LFFANet, self).__init__()

        self.in_channels = in_channels
        self.num=1
        self.backbone = LISDNet_backbone(self.in_channels)
        filters = [8, 16, 32, 64, 128]
        self.dropout = nn.Dropout(0.1)

        ## -------------Encoder--------------
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        #self.backbone= cefenzhi_ghostbackbone_n (self.in_channels)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks  # UNet3 paper : 64 x N(CatBlocks) / In my paper (AMFU-net) : 8 x N(CatBlocks)

        '''stage 4d'''
        # h1->256*256, hd4->32*32, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = CONV(filters[0], self.CatChannels, 3, s=1,p=1)


        # h2->128*128, hd4->32*32, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = CONV(filters[1], self.CatChannels, 3, s=1,p=1)

        # h3->64*64, hd4->32*32, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = CONV(filters[2], self.CatChannels, 3, 1,1)

        # h4->32*32, hd4->32*32, Concatenation
        self.h4_Cat_hd4_conv = CONV(filters[3], self.CatChannels, 3, 1,1)

        # hd5->16*16, hd4->32*32, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = CONV(filters[4], self.CatChannels, 3,1,1)


        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = CONV(self.UpChannels, self.UpChannels, 3, 1,1)


        '''stage 3d'''
        # h1->256*256, hd3->64*64, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = CONV(filters[0], self.CatChannels, 3,1,1)


        # h2->128*128, hd3->64*64, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = CONV(filters[1], self.CatChannels, 3, 1,1)


        # h3->64*64, hd3->64*64, Concatenation
        self.h3_Cat_hd3_conv = CONV(filters[2], self.CatChannels, 3, 1,1)

        # hd4->32*32, hd4->64*64, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv =CONV(self.UpChannels, self.CatChannels, 3, 1,1)

        #####
        #self.hd4_conv=seCBAM_block(128+64+8, filters[3])
        self.hd4_conv=self.DACM(128+64+8, filters[3],self.num)
        #self.hd4_conv=Res_CBAM_block(128+64+8, filters[3])
        self.hd4_UT_hd3_conv2 = CONV(filters[3], self.CatChannels, 3, 1,1)
        #####


        # hd5->16*16, hd4->64*64, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = CONV(filters[4], self.CatChannels, 3,1,1)


        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = CONV(self.UpChannels, self.UpChannels, 3, 1,1)


        '''stage 2d '''
        # h1->256*256, hd2->128*128, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = CONV(filters[0], self.CatChannels, 3, 1,1)


        # h2->128*128, hd2->128*128, Concatenation
        self.h2_Cat_hd2_conv = CONV(filters[1], self.CatChannels, 3, 1,1)


        # hd3->64*64, hd2->128*128, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = CONV(self.UpChannels, self.CatChannels, 3, 1,1)

        #####
        self.hd3_UT_hd2_conv2 = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        #self.hd3_conv=seCBAM_block(64+32+8,filters[2])
        self.hd3_conv = self.DACM(64+32+8,filters[2], self.num)
        #self.hd3_conv=Res_CBAM_block(64+32+8,filters[2])
        #####

        # hd4->32*32, hd2->128*128, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = CONV(self.UpChannels, self.CatChannels, 3, 1,1)


        # hd5->16*16, hd2->128*128, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = CONV(filters[4], self.CatChannels, 3, 1,1)


        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = CONV(self.UpChannels, self.UpChannels, 3, 1,1)


        '''stage 1d'''
        # h1->256*256, hd1->256*256, Concatenation
        self.h1_Cat_hd1_conv = CONV(filters[0], self.CatChannels, 3, 1,1)


        # hd2->128*128, hd1->256*256, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv =CONV(self.UpChannels, self.CatChannels, 3, 1,1)

        #####
        #self.hd2_conv=seCBAM_block(16+32+8,filters[1])
        self.hd2_conv = self.DACM(16+32+8,filters[1], self.num)
        #self.hd2_conv=Res_CBAM_block(16+32+8,filters[1])
        self.hd2_UT_hd1_conv2 = CONV(filters[1], self.CatChannels, 3, 1,1)
        #####

        # hd3->64*64, hd1->256*256, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd3_UT_hd1_conv = CONV(self.UpChannels, self.CatChannels, 3, 1,1)


        # hd4->32*32, hd1->256*256, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd4_UT_hd1_conv = CONV(self.UpChannels, self.CatChannels, 3, 1,1)


        # hd5->16*16, hd1->256*256, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = CONV(filters[4], self.CatChannels, 3, 1,1)


        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 =CONV(self.UpChannels, self.UpChannels, 3, 1,1)


        # -------------Bilinear Upsampling--------------
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], 1, 3, padding=1)

        # Cat attention
        #self.cat_attn = seCBAM_block(48, self.UpChannels)
        self.cat_attn = self.DACM(48, self.UpChannels,self.num)

        #self.cat_attn1 = seCBAM_block(40, self.UpChannels)
        self.cat_attn1 = self.DACM(40, self.UpChannels, self.num)


        self.final_conv=nn.Conv2d(5,1,1,1,0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                my_init_weights(m, init_type='xavier')
            elif isinstance(m, nn.BatchNorm2d):
                my_init_weights(m, init_type='xavier')

    def DACM(self, inch,outch,num):
        layers=[]
        layers.append(seCBAM_block(inch, outch))
        for i in range(num-1):
            layers.append(seCBAM_block(outch,outch))
        return nn.Sequential(*layers)
    def forward(self, inputs):
        # -------------Encoder-------------
        h1,h2,h3,h4,hd5=self.backbone(inputs)

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))

        hd5_UT_hd4 = self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))
        #####
        h4_temp = self.hd4_conv(torch.cat([h4, self.hd5_UT_hd4(hd5), self.maxpool3(h3_Cat_hd3)], 1))
        h4_cat_h4 = self.hd4_UT_hd3_conv2(h4_temp)
        # print('h4_cat_h4.shape:')
        # print(h4_cat_h4.shape)
        # print('h4_temp.shape:')
        # print(h4_temp.shape)
        #####
        hd4 = self.conv4d_1(
            self.cat_attn(torch.cat((h4_cat_h4,h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4),
                                    1)))  # hd4_cat->[16, 40, 32, 32] [B, C, W, H] / filter 8

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))

        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))

        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))
        #####
        h3_temp = self.hd3_conv(torch.cat([h3, self.hd4_UT_hd3(h4_temp), self.maxpool2(h2_Cat_hd2)], 1))
        hd3_cat_hd3 = self.hd3_UT_hd2_conv2(h3_temp)
        # print('hd3_cat_hd3.shape:')
        # print(hd3_cat_hd3.shape)
        # print('h3_temp.shape:')
        # print(h3_temp.shape)
        #####
        hd3 = self.conv3d_1(
            self.cat_attn(torch.cat((hd3_cat_hd3,h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3),
                                    1))) # hd3_cat->[16, 40, 64, 64] [B, C, W, H] / filter 8

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))

        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))

        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))
        #####
        # h1_pt_h2_temp
        h2_temp = self.hd2_conv(torch.cat([h2, self.hd3_UT_hd2(h3_temp), self.maxpool1(h1_Cat_hd1)], 1))
        hd2_cat_hd2 = self.hd2_UT_hd1_conv2(h2_temp)
        # print('hd2_cat_hd2.shape:')
        # print(hd2_cat_hd2.shape)
        # print('h2_temp.shape:')
        # print(h2_temp.shape)
        #####
        hd2 = self.conv2d_1(
            self.cat_attn(torch.cat((hd2_cat_hd2,h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2),
                                    1)))  # hd2_cat->[16, 40, 128, 128] [B, C, W, H] / filter 8


        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))

        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))
        # print(h1_Cat_hd1.shape)
        # print(hd2_UT_hd1.shape)
        # print(hd3_UT_hd1.shape)
        # print(hd4_UT_hd1.shape)
        # print(hd5_UT_hd1.shape)
        hd1 = self.conv1d_1(
            self.cat_attn1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1),
                                    1)))  # hd1_cat->[16, 40, 256, 256] [B, C, W, H] / filter 8

        # DSV (DeepSuperVision)
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256  (256*256*1)

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256  (256*256*1)

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256  (256*256*1)

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256 (256*256*1)

        d1 = self.outconv1(hd1)  # 256      (256*256*1)

        fused_feature=self.final_conv(torch.cat([d1,d2,d3,d4,d5],1))

        return [d5, d4, d3, d2, d1, fused_feature]


if __name__ == "__main__":
    import time

    x = torch.rand(4, 3, 256, 256)
    start_time = time.time()  # 程序开始时间
    net = LISDNet()
    net = net.cuda()
    x = x.cuda()
    output = net(x)
    end_time = time.time()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print(run_time)
    print(output.shape)