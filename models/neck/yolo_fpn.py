import torch
import torch.nn as nn
# import torch.nn.functional as F
# import sys

# from models.ops import ConvBNLayer
from models.backbone.csp_darknet import BaseConv, CSPLayer, DWConv



# from ops import ConvBNLayer


# from backbone.csp_darknet import BaseConv, CSPLayer, DWConv



class YOLOXPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, blocks):
        assert len(blocks) == len(self.in_channels)
        [x2, x1, x0] = blocks

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs


if __name__ == "__main__":
    # from thop import profile

    in_channels = [96, 192, 384]
    # feats = [torch.rand([1, in_channels[0], 64, 64]), 
    #          torch.rand([1, in_channels[1], 32, 32]),
    #          torch.rand([1, in_channels[2], 16, 16])]

    feats = [torch.rand([1, in_channels[0], 80, 80]), 
             torch.rand([1, in_channels[1], 40, 40]),
             torch.rand([1, in_channels[2], 20, 20])]


    fpn = YOLOXPAFPN(depth=0.33, width=0.375)
    # fpn.init_weights()
    # print(fpn)
    # fpn.eval()
    # total_ops, total_params = profile(fpn, (feats,))
    # print("total_ops {:.2f}G, total_params {:.2f}M".format(total_ops/1e9, total_params/1e6))
    output = fpn(feats)
    for o in output:
        print(o.size())
