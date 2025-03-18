import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from utils.Lightening import RadiomicsLSTMClassifier
from utils.new_dataloader import RadiomicsDataset
import pandas as pd
from sklearn.model_selection import train_test_split

# TODO: Relabel the data. 0,1 --> Response, 2,3, 4 --> No Response
import torch


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 10
    num_workers = 4
    max_epochs = 50
    min_epochs = 1
    check_val_every_n_epoch = 3

    features_file = "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\radiomics_features.xlsx"
    db_file = "C:\\Users\\aaron.l\\Documents\\db_20241213.xlsx"
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    df = pd.read_excel(features_file)

    db = pd.read_excel(db_file, usecols=["cnda_subject_label", "AJCC Stage grouping "])
    db = db.dropna(
        subset=["AJCC Stage grouping "]
    )  # Remove patients with null responses
    db["Response"] = db["AJCC Stage grouping "].apply(
        lambda x: 0 if x in {0, 1, 2} else 1
    )
    db = db.drop_duplicates(subset=["cnda_subject_label"])  # Drop duplicate patients

    db["Patient_ID"] = db["cnda_subject_label"].str.replace("cnda_", "", regex=False)

    df = df.merge(db[["Patient_ID", "Response"]], on="Patient_ID")

    grouped = df.groupby("Patient_ID").filter(
        lambda x: set(x["Session"]) == {"MR1", "MR2"}
    )

    train_df, test_df = train_test_split(
        grouped, test_size=0.25, stratify=grouped["Response"], random_state=42
    )

    train_dataset = RadiomicsDataset(train_df)
    test_dataset = RadiomicsDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Callbacks
    early_stopping = EarlyStopping(monitor="validation/loss", patience=40, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss",
        dirpath="checkpoints/",
        filename="best-checkpoint-{epoch:02d}-{validation/loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    model = RadiomicsLSTMClassifier()

    # Initialize WandB Logger
    run_name = f"radiomics_lstm_prediction_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch={batch_size}"
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
        log_every_n_steps=1,
    )

    # Train and Validate
    trainer.fit(model, train_loader, test_loader)
    trainer.validate(model, test_loader)

    # Finish WandB
    wandb_logger.experiment.unwatch(model)
    wandb.finish()


if __name__ == "__main__":
    main()
