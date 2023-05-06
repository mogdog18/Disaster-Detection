import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class Encoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.encoder = sam.image_encoder

        if freeze:
            self._freeze_wts()

    def _freeze_wts(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pre, post):
        x1 = self.encoder(pre)
        x2 = self.encoder(post)
        x = torch.cat([x1, x2], dim=1)
        return x
