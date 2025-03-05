import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import wandb


class LSTMTimeSeriesClassifier(pl.LightningModule):
    def __init__(self, hidden_size=128, num_layers=2, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()  # Logs hyperparameters for WandB

        # Define LSTM
        self.lstm = nn.LSTM(
            input_size=256 * 256,  # Each MRI scan is flattened to 65536
            hidden_size=hidden_size,  # Default = 128
            num_layers=num_layers,  # Default = 2
            batch_first=True,
        )

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_size, 1)

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x1, x2):
        batch_size, channel, H, W = x1.shape  # x1 and x2 are both (B, 256, 256)
        x = torch.stack([x1, x2], dim=1).view(batch_size, 2, -1)
        lstm_out, _ = self.lstm(x)  # LSTM processes sequence
        lstm_last_step = lstm_out[:, -1, :]  # Take last step output
        logits = self.fc(lstm_last_step)  # Classification output
        return logits.squeeze(1)

    def training_step(self, batch, batch_idx):
        loss, acc, auroc = self.common_step(batch, batch_idx)
        # Log loss & metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_epoch=True)

        return loss

    def common_step(self, batch, batch_idx):
        x1, x2, y = batch  # Unpack batch
        logits = self(x1, x2)  # Forward pass
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        preds = probs > 0.5  # Binary predictions for accuracy
        acc = self.accuracy(preds, y)
        auroc = self.auroc(probs, y)  # Pass probabilities, not binary values
        return loss, acc, auroc

    def validation_step(self, batch, batch_idx):
        loss, acc, auroc = self.common_step(batch, batch_idx)
        return {"val_loss": loss, "val_acc": acc, "val_auroc": auroc}

    def validation_epoch_end(self, outputs):
        # Aggregate the results from the validation_step
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_auroc = torch.stack([x["val_auroc"] for x in outputs]).mean()

        # Log the average metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
        self.log("val_auroc", avg_auroc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
