"""
UNet
The main UNet model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





# Utility Functions
''' when filter kernel= 3x3, padding=1 makes in&out matrix same size'''
def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )


def merge_cbl(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        # nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True)
    )



def down_pooling():
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# UNet class
class AttentionGate(nn.Module):
    """
    To build an attention gate block which can be difined as: h_output = alpha * h, where h_output is the output of
    the gate; h is one of the input of the gate; alpha is the computed attention coefficients, which can be defined as:
    alpha = sigma2 * { Wk * [ Wint * ( sigma1 * (Wh * h + Wg * g + b_h,g)) + b_int] + b_k}, where g and h represent the
    feature maps presented to the inputs of AG modules from the decoder and encoder piplines and Wg, Wh, Wint, Wk
    indicate the convolution kernels. Furthermore, sigma1 is the Relu activation function and sigma2 is the Sigmoid
    activation function.
    """
    def __init__(self, in_channels_g, in_channels_h, in_channels_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, in_channels_int, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_channels_int)
        )
        self.W_h = nn.Sequential(
            nn.Conv2d(in_channels_h, in_channels_int, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_channels_int)
        )
        self.alpha = nn.Sequential(
            nn.Conv2d(in_channels_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, h):
        g1 = self.W_g(g)
        h1 = self.W_h(h)
        alpha = self.relu(g1 + h1)
        alpha = self.alpha(alpha)
        return h * alpha


class UNet2(nn.Module):
    def __init__(self, input_channels, nclasses):
        super().__init__()
        # go down
        self.conv1 = conv_bn_leru(1,64)
        self.conv2 = conv_bn_relu(64, 128)
        self.Res2 = nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0)
        self.conv3 = conv_bn_relu(128, 256)
        self.Res3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = conv_bn_relu(256, 512)
        self.Res4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv5 = conv_bn_relu(512, 1024)
        self.Res5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.down_pooling = nn.MaxPool2d(2)
        self.Mulchannel = input_channels

        # go up
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.Res6 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.Res7 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.Res8 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.Res9 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        self.conv10 = nn.Conv2d(64, nclasses, 1)
        self.AG1 = AttentionGate(512,512,512)
        self.AG2 = AttentionGate(256,256,256)
        self.AG3 = AttentionGate(128,128,128)
        self.AG4 = AttentionGate(64,64,64)


        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        # normalize input data
        x = x/255.
        if self.Mulchannel == 1:
            #x = 0 #mul_test
            x = x[:,1,:,:] #t2
            x = x[:,np.newaxis,:,:]
        else:
            x = 0   #check

        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = torch.add(self.conv2(p1),self.Res2(p1))
        p2 = self.down_pooling(x2)
        x3 = torch.add(self.conv3(p2),self.Res3(p2))
        p3 = self.down_pooling(x3)
        x4 = torch.add(self.conv4(p3),self.Res4(p3))
        p4 = self.down_pooling(x4)
        x5 = torch.add(self.conv5(p4),self.Res5(p4))

        # go up
        p6 = self.up_pool6(x5)
        A6 = self.AG1(p6,x4)
        x6 = torch.cat([p6, A6], dim=1)        
        x6 = torch.add(self.conv6(x6),self.Res6(x6))
        p7 = self.up_pool7(x6)
        A7 = self.AG2(p7,x3)
        x7 = torch.cat([p7, A7], dim=1)
        x7 = torch.add(self.conv7(x7),self.Res7(x7))

        p8 = self.up_pool8(x7)
        A8 = self.AG3(p8,x2)
        x8 = torch.cat([p8, A8], dim=1)
        x8 = torch.add(self.conv8(x8),self.Res8(x8))


        p9 = self.up_pool9(x8)
        A9 = self.AG4(p9,x1)
        x9 = torch.cat([p9, A9], dim=1)
        x9 = torch.add(self.conv9(x9),self.Res9(x9))



        output = self.conv10(x9)
        output = torch.sigmoid(output)
        # output = torch.relu()

        return output


class MultiUNet(nn.Module):
    def __init__(self, input_channels, nclasses):
        super().__init__()
        # go down
        self.conv1 = conv_bn_leru(input_channels,16)
        self.mer_conv1 = merge_cbl(32,16)

        self.conv2 = conv_bn_leru(16, 32)
        self.mer_conv2 = merge_cbl(64,32)

        self.conv3 = conv_bn_leru(32, 64)
        self.mer_conv3 = merge_cbl(128,64)

        self.merge1 = merge_cbl(128,64)

        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.down_pooling = nn.MaxPool2d(2)
        self.merge1 = merge_cbl(128,64)
        self.merge2 = merge_cbl(256,128)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_leru(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_leru(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_leru(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_leru(32, 16)

        self.conv10 = nn.Conv2d(16, nclasses, 1)


        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        # normalize input data
        x = x/255.
        dwi = x[:,0,:,:]#0-dwi,1-t1,2-t2
        t1 = x[:,1,:,:]
        t2 = x[:,2,:,:]
        dwi = dwi[:,np.newaxis,:,:]
        t1 = t1[:, np.newaxis, :, :]
        t2 = t2[:, np.newaxis, :, :]

        dwi_1 = self.conv1(dwi)                       ##16
        t1_1 = self.conv1(t1)
        t2_1 = self.conv1(t2)

        merge1_1 = torch.cat([dwi_1,t1_1],dim=1)
        merge1_2 = self.mer_conv1(merge1_1)
        merge1_3 = torch.cat([merge1_2,t2_1],dim=1)
        merge1_4 = self.mer_conv1(merge1_3)            #can be used 16

        dwi_p1 = self.down_pooling(dwi_1)
        t1_p1 = self.down_pooling(t1_1)
        t2_p1 = self.down_pooling(t2_1)

        dwi_2 = self.conv2(dwi_p1)                   ##32
        t1_2 = self.conv2(t1_p1)
        t2_2 = self.conv2(t2_p1)

        merge2_1 = torch.cat([dwi_2, t1_2], dim=1)
        merge2_2 = self.mer_conv2(merge2_1)              #32
        merge2_3 = torch.cat([merge2_2, t2_2], dim=1)
        merge2_4 = self.mer_conv2(merge2_3)             # can be used

        dwi_p2 = self.down_pooling(dwi_2)
        t1_p2 = self.down_pooling(t1_2)
        t2_p2 = self.down_pooling(t2_2)

        dwi_3 = self.conv3(dwi_p2)                        ##64
        t1_3 = self.conv3(t1_p2)
        t2_3 = self.conv3(t2_p2)

        merge3_1 = torch.cat([dwi_3, t1_3], dim=1)
        merge3_2 = self.mer_conv3(merge3_1)              #64
        merge3_3 = torch.cat([merge3_2, t2_3], dim=1)
        merge3_4 = self.mer_conv3(merge3_3)              # can be used

        merge1 = torch.cat([t1_3,t2_3],dim=1)
        merge2 = self.merge1(merge1)                #64

        dwi_p3 = self.down_pooling(dwi_3)
        merge3 = self.down_pooling(merge2)

        dwi_4 = self.conv4(dwi_p3)              #128
        merge4 = self.conv4(merge3)
        # print(merge4.shape)
        # print(dwi_4.shape)

        merge4 = torch.cat([dwi_4,merge4],dim=1)  #256
        # print(merge4.shape)
        merge5 = self.merge2(merge4)          #128

        merge6 = self.down_pooling(merge5)
        merge7 = self.conv5(merge6)          ##256









        # x1 = self.conv1(t2)
        # p1 = self.down_pooling(x1)
        # x2 = self.conv2(p1)
        # p2 = self.down_pooling(x2)
        # x3 = self.conv3(p2)
        # p3 = self.down_pooling(x3)
        # x4 = self.conv4(p3)
        # p4 = self.down_pooling(x4)
        # x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(merge7)
        x6 = torch.cat([p6, merge5], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7,merge3_4], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, merge2_4], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, merge1_4], dim=1)
        x9 = self.conv9(x9)


        output = self.conv10(x9)
        output = torch.sigmoid(output)
        # output = torch.relu()

        return output


