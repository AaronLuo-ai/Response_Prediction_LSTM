import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import torchvision.transforms as T
from utils.dataloader import TwoPartDataset
from utils.Lightening import LSTMTimeSeriesClassifier


def main():

    batch_size = 10
    num_workers = 4
    max_epochs = 50
    min_epochs = 1
    check_val_every_n_epoch = 5

    train_transform = T.Compose(
        [
            T.CenterCrop((256, 256)),  # Crop to (256, 256)
            T.RandomRotation(degrees=15),  # Randomly rotate ±15 degrees
            T.Normalize(
                mean=[0.5], std=[0.5]
            ),  # Normalize (replace with computed values)
        ]
    )

    test_transform = T.Compose(
        [
            T.CenterCrop((256, 256)),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    TrainDataset = TwoPartDataset(phase="train", transforms=train_transform)
    TestDataset = TwoPartDataset(phase="test", transforms=test_transform)

    TrainLoader = DataLoader(
        TrainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    TestLoader = DataLoader(
        TestDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Initialize Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=40, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-checkpoint-{epoch:02d}-{validation/loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    model = LSTMTimeSeriesClassifier()

    # Initialize WandB Logger
    run_name = f"lstm_prediction_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch={batch_size}"
    wandb_logger = WandbLogger(
        log_model=False, project="LSTM-Prediction", name=run_name
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    # Train and Validate
    trainer.fit(model, TrainLoader, TestLoader)
    trainer.validate(model, TestLoader)

    # Finish WandB
    wandb_logger.experiment.unwatch(model)
    wandb.finish()


if __name__ == "__main__":
    main()
