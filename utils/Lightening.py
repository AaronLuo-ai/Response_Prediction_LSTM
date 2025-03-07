import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import wandb
from torchmetrics.classification import BinaryAccuracy
import torchvision.utils as vutils


class LSTMTimeSeriesClassifier(pl.LightningModule):
    def __init__(self, hidden_size=128, num_layers=2, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()  # Logs hyperparameters for WandB
        self.lr = lr
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
        self.loss_fn = nn.MSELoss()  # More stable than BCELoss
        self.accuracy_metric = BinaryAccuracy()  # Accuracy metric using TorchMetrics
        self.auc_metric = torchmetrics.AUROC(task="binary")

    def on_train_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_validation_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_test_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def forward(self, x1, x2):
        batch_size, channel, H, W = x1.shape  # x1 and x2 are both (B, 256, 256)
        x = torch.stack([x1, x2], dim=1).view(batch_size, 2, -1)
        lstm_out, _ = self.lstm(x)  # LSTM processes sequence
        lstm_last_step = lstm_out[:, -1, :]  # Take last step output
        logits = self.fc(lstm_last_step)  # Classification output
        return logits.squeeze(1)

    def _common_step(self, batch, batch_idx):
        x1, x2, y = batch
        scores = self.forward(x1, x2)
        loss = self.loss_fn(
            scores, y.float()
        )  # Ensure labels are float for BCEWithLogitsLoss
        return loss, scores, y, x1, x2

    def training_step(self, batch, batch_idx):
        loss, scores, y, _, _ = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)  # Convert logits to probabilities

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y, x1, x2 = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
            grid_input1 = vutils.make_grid(x1[:8], normalize=True, scale_each=True)
            grid_input2 = vutils.make_grid(x2[:8], normalize=True, scale_each=True)
            self.logger.experiment.log(
                {
                    "validation/input_images1": wandb.Image(
                        grid_input1, caption="Input Images1"
                    ),
                    "validation/input_images2": wandb.Image(
                        grid_input2, caption="Input Images2"
                    ),
                    "global_step": self.trainer.global_step,
                }
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, _, _ = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("train/auc", self.auc_metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("validation/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("validation/auc", self.auc_metric.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
