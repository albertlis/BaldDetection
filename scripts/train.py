from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

from models.resnet50 import ResNetModel
from src.celeba_dataset import get_dataloaders


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
    main()
