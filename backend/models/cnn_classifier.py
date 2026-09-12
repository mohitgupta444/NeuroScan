"""
CNN Classifier for Brain Tumor MRI Classification
---------------------------------------------------
Backbone: MobileNetV2 (pretrained on ImageNet, fine-tuned here)
Why MobileNetV2: it is small (~3.5M params) and fast on CPU / integrated
graphics, which fits an 8GB RAM laptop with no dedicated GPU.

Classes: glioma, meningioma, notumor, pituitary
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TumorCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        # Load MobileNetV2 backbone. We keep the feature extractor
        # (the "encoder") and replace the final classification head.
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # backbone.features is the convolutional encoder -- this is the
        # SAME encoder we reuse inside the U-Net (see unet_model.py) so we
        # only train one heavy feature extractor, not two.
        self.encoder = backbone.features  # output: [B, 1280, H/32, W/32]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features: bool = False):
        feats = self.encoder(x)          # [B, 1280, H/32, W/32]
        pooled = self.pool(feats).flatten(1)  # [B, 1280]
        logits = self.classifier(pooled)

        if return_features:
            # Returned for Grad-CAM (pseudo-mask generation) and for the
            # U-Net decoder to reuse as skip-connection-free deep features.
            return logits, feats
        return logits


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
