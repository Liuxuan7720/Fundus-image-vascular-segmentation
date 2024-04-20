import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
class HDD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HDD, self).__init__()

        self.in_ch = in_ch
        self.mid_mid = out_ch // 4
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])


        self.conv3x3_3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])

        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.rel = nn.ReLU(inplace=True)
        if self.in_ch != self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]

        x4 = self.conv3x3_1_1(x1)
        x5 = self.conv3x3_2_1(x2 + x4)
        x6 = self.conv3x3_3_1(x5 + x3)
        xxx = self.conv1x1_1(torch.cat((x0, x4, x5, x6), dim=1))
        x0_0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]

        x4 = self.conv3x3_1_2(x1_2)
        x5 = self.conv3x3_2_2(x4 + x2_2)
        x6 = self.conv3x3_3_2(x5 + x3_2)
        xx = torch.cat((x0_0, x4, x5, x6), dim=1)
        xx = self.conv1x1_2(xx)
        if self.in_ch != self.out_ch:
            x = self.short_connect(x)
        return self.rel(xx + x + xxx)
class HDD_Net(nn.Module):

    def __init__(self):

        super(HDD_Net, self).__init__()

        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            HDD(32,32),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = HDD(32,64)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = HDD(64,128)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = HDD(128,256)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = HDD(256,512)

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = HDD(512,256)

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = HDD(256,128)

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = HDD(128,64)

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = HDD(64,32)

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=2,
                                    kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        #print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        #print('after conv1', x.shape)
        conv1_out = x  # Save out1
        x = self.max1(x)
        #print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        #print('after conv2', x.shape)
        conv2_out = x

        x = self.max2(x)
        #print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        #print('after conv3', x.shape)
        conv3_out = x

        x = self.max3(x)
        #print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        #print('after conv5', x.shape)
        conv4_out = x

        x = self.max4(x)

        # Midpoint
        x = self.conv5_block(x)

        # Up 1
        x = self.up_1(x)
        #print('up_1', x.shape)
        #print('conv4_out', conv4_out.shape)
        x = torch.cat([x, conv4_out], dim=1)
        # #print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # #print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # #print('up_2', x.shape)

        x = torch.cat([x, conv3_out], dim=1)
        # #print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # #print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # #print('up_3', x.shape)

        x = torch.cat([x, conv2_out], dim=1)
        # #print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # #print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # #print('up_4', x.shape)

        x = torch.cat([x, conv1_out], dim=1)
        # #print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # #print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        return x


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 572, 572)
    model = CleanU_Net()
    x = model(im)
    # #print(x.shape)
    del model
    del x
    # #print(x.shape)
