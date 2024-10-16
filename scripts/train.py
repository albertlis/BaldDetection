from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

from models.resnet50 import ResNetModel
from src.celeba_dataset import get_dataloaders

def set_best_performance():
    """Disable in case of issues"""
    # use only with constant size images:
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # If FP16 is faster than FP32 try this:
    # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
    # and torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("medium")



@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    train_loader, val_loader, test_loader = get_dataloaders(
        Path(cfg.dataset.images_dir_path),
        Path(cfg.dataset.csv_attributes_path),
        cfg.training.batch_size
    )

    # Logger
    logger = TensorBoardLogger("logs", name="bald_detection")

    # Trener
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        logger=logger,
        devices=cfg.training.gpus
    )

    training_steps = len(train_loader)
    # Inicjalizacja modelu
    model = ResNetModel(learning_rate=cfg.model.learning_rate, training_steps=training_steps)

    # Trening
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    set_best_performance()
    main()
