import torch
import torch.nn as nn
import torch.nn.functional as func

# Largely stolen from https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py


class FPNBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(FPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        # Unlike residual blocks, the skip connect in FPN is added to the final layer rather than concatenated, so we
        # need to make sure the skip connect has the correct number of channels
        self.skip_connect = nn.Sequential(
            nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels))

    def forward(self, x):
        out = func.relu(self.bn1(self.conv1(x)))
        out = func.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.skip_connect(x)
        return func.relu(out)


class FPN(nn.Module):

    def __init__(self, in_channels, num_blocks):
        super(FPN, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self.generate_blocks(num_blocks[0], 64, stride=1)
        self.layer2 = self.generate_blocks(num_blocks[1], 128, stride=2)
        self.layer3 = self.generate_blocks(num_blocks[2], 256, stride=2)
        self.layer4 = self.generate_blocks(num_blocks[3], 512, stride=2)

        # Top layer
        self.top = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smoothing
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral connections
        self.lateral1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def generate_blocks(self, num_blocks, out_channels, stride):
        block_strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in block_strides:
            layers.append(FPNBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * FPNBlock.expansion
        return nn.Sequential(*layers)

    def upsample_add(self, x, y):
        """
        Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return func.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

    def forward(self, x):
        # Bottom-up
        c1 = func.relu(self.bn1(self.conv1(x)))
        c1 = func.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top down
        p5 = self.top(c5)
        p4 = self.upsample_add(p5, self.lateral1(c4))
        p3 = self.upsample_add(p4, self.lateral2(c3))
        p2 = self.upsample_add(p3, self.lateral3(c2))

        # Smoothing
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


def main():
    network = FPN(64, [2, 2, 2, 2])
    input = torch.randn(1, 3, 600, 900)
    output = network(input)
    for key, value in output.items():
        print("{}: {}".format(key, value.shape))


if __name__ == "__main__":
    main()