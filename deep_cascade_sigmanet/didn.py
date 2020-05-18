"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
#import net_utils

class _Residual_Block(nn.Module):
    def __init__(self, num_chans=64):
        super(_Residual_Block, self).__init__()
        bias = True
        #res1
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(num_chans, num_chans * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(num_chans * 4, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(num_chans * 4, num_chans * 8, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(num_chans * 4, num_chans * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        #res4
        self.conv16 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(num_chans * 2, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        #res5
        self.conv21 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self, num_chans=64):
        super(Recon_Block, self).__init__()
        bias=True
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output



class DIDN(nn.Module):
    """
    Deep Iterative Down-Up Network, NTIRE denoising challenge winning entry

    Source: http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdfp

    """
    def __init__(self, in_chans=2, out_chans=2, num_chans=64,
                 pad_data=False, global_residual=False, n_res_blocks=6):
        super().__init__()
        self.pad_data = pad_data
        self.global_residual = global_residual
        bias=True
        self.conv_input = nn.Conv2d(in_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu2 = nn.PReLU()

        self.n_res_blocks = n_res_blocks
        recursive = []
        for i in range(self.n_res_blocks):
            recursive.append(_Residual_Block(num_chans))
        self.recursive = torch.nn.ModuleList(recursive)

        self.conv_mid = nn.Conv2d(num_chans * self.n_res_blocks, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(num_chans // 4, out_chans, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):

        x = x.permute(0,3,1,2)

        if self.pad_data:
            orig_shape2d = x.shape[-2:]
            p2d = net_utils.calculate_downsampling_padding2d(x, 3)
            x = net_utils.pad2d(x, p2d)

        residual = x
        out = self.relu1(self.conv_input(x))
        out = self.relu2(self.conv_down(out))

        recons = []
        for i in range(self.n_res_blocks):
            out = self.recursive[i](out)
            recons.append(out)

        out = torch.cat(recons, 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)

        if self.global_residual:
            out = torch.add(out, residual)

        if self.pad_data:
            out = net_utils.unpad2d(out, orig_shape2d)

        out = out.permute(0, 2, 3, 1)

        return out
