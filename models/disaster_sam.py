from models.encoder_sam import Encoder
from models.simple_decoder import DecoderV1, DecoderV2
import torch.nn as nn


class DisasterSamV1(nn.Module):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(freeze=freeze_encoder)
        self.decoder = DecoderV1(num_classes)

    def forward(self, pre, post):
        x = self.encoder(pre, post)
        x = self.decoder(x)
        return x


class DisasterSamV2(nn.Module):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(freeze=freeze_encoder)
        self.decoder = DecoderV2(in_channels=512, num_classes=num_classes)

    def forward(self, pre, post):
        x = self.encoder(pre, post)
        x = self.decoder(x)
        return x
