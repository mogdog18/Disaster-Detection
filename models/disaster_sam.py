from models.encoder_sam import Encoder
from models.simple_decoder import Decoder
import torch.nn as nn


class DisasterNet(nn.Module):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(freeze=freeze_encoder)
        self.decoder = Decoder(num_classes)

    def forward(self, pre, post):
        x = self.encoder(pre, post)
        x = self.decoder(x)
        return x
