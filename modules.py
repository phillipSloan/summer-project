import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm=nn.BatchNorm3d, kernel_size=3, double=True, skip=True):
        super().__init__()
        self.skip = skip
        self.downsample = in_planes != out_planes
        self.final_activation = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        padding = (kernel_size - 1) // 2
        if double:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                nn.BatchNorm3d(out_planes),
                nn.Dropout3d(p=0.25),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                nn.BatchNorm3d(out_planes))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=padding),
                nn.BatchNorm3d(out_planes),
                nn.Dropout3d(p=0.25))

        if self.skip and self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1,
                          padding=0),
                norm(out_planes))

    def forward(self, x):
        y = self.conv_block(x)
        if self.skip:
            res = x
            if self.downsample:
                res = self.conv_down(res)
            y = y + res
        return self.final_activation(y)


# green block in Fig.1
class TranspConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2,
                                        padding=0, output_padding=0,bias=False)

    def forward(self, x):
        y = self.block(x)
        return y


class SkipBlock(nn.Module):
    def __init__(self, in_planes, out_planes, layers=1, conv_block=False):
        super().__init__()
        self.blocks = nn.ModuleList([TranspConv3DBlock(in_planes, out_planes),
                                            ])
        if conv_block:
            self.blocks.append(Conv3DBlock(out_planes, out_planes, double=False))

        if int(layers)>=2:
            for _ in range(int(layers) - 1):
                self.blocks.append(TranspConv3DBlock(out_planes, out_planes))
                if conv_block:
                    self.blocks.append(Conv3DBlock(out_planes, out_planes, double=False))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
