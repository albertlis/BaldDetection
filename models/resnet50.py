from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import pytorch_lightning as pl


class ResNetModel(pl.LightningModule):
    def __init__(self, learning_rate: float, training_steps: int):
        super(ResNetModel, self).__init__()
        self.save_hyperparameters()
        model = models.resnet50(pretrained=True)

        self.model = nn.Sequential()
        for name, child in model.named_children():
            self.model.append(child)
            if "layer4" in name:
                break

        self.learning_rate = learning_rate
        self.training_steps = training_steps

    def forward(self, x):
        x = self.model(x)
        return torch.mean(x, dim=(1, 2, 3))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='mean')
        self.log('train_loss', loss)
        return loss

    #TODO
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.training_steps, eta_min=5e-5)
        return [optimizer], [scheduler]
