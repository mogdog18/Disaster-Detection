import torch.nn.functional as F
import torch.nn as nn


class DecoderV1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        super(DecoderV1, self).__init__()
        self.combine = nn.Conv2d(2 * 256, 256, 1, 1)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.combine(x)
        x = F.relu(self.conv1(self.upconv1(x)))
        x = F.relu(self.conv2(self.upconv2(x)))
        x = F.relu(self.conv3(self.upconv3(x)))

        x = self.upconv4(x)

        return F.softmax(x, dim=1)


class DecoderV2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        super(DecoderV2, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        in_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        in_channels = in_channels // 2
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        in_channels = in_channels // 2
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels, num_classes, kernel_size=2, stride=2
        )

    def forward(self, x):
        x = F.relu(self.conv1(self.upconv1(x)))
        x = F.relu(self.conv2(self.upconv2(x)))
        x = F.relu(self.conv3(self.upconv3(x)))
        x = self.upconv4(x)

        return F.softmax(x, dim=1)
