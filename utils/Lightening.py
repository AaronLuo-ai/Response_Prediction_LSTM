from sklearn.svm import SVC
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
import torch
from torchmetrics.classification import BinaryAccuracy
import torchmetrics

encoder_weight_path = "C:\\Users\\aaron.l\\Documents\\checkpoints_segModel\\best-checkpoint-  epoch=92-validation\\loss=0.0054.ckpt"


class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 8, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(4)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        enc = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        enc = self.pool2(self.relu2(self.bn2(self.conv2(enc))))
        enc = self.pool3(self.relu3(self.bn3(self.conv3(enc))))
        enc = self.relu4(self.bn4(self.conv4(enc)))
        enc = self.relu5(self.bn5(self.conv5(enc)))
        enc = self.conv6(enc)
        return enc


class MyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder using ConvTranspose2d
        self.deconv0 = nn.ConvTranspose2d(
            4, 128, kernel_size=4, stride=2, padding=1
        )  # Upsample to 8x8
        self.dbn0 = nn.BatchNorm2d(128)
        self.drelu0 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # Upsample to 16x16
        self.dbn1 = nn.BatchNorm2d(64)
        self.drelu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # Upsample to 32x32
        self.dbn2 = nn.BatchNorm2d(32)
        self.drelu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )  # Upsample to 64x64
        self.dbn3 = nn.BatchNorm2d(16)
        self.drelu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(
            16, 8, kernel_size=4, stride=2, padding=1
        )  # Upsample to 128x128
        self.dbn4 = nn.BatchNorm2d(8)
        self.drelu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(
            8, 1, kernel_size=4, stride=2, padding=1
        )  # Upsample to 256x256

    def forward(self, enc):
        dec = self.drelu0(self.dbn0(self.deconv0(enc)))
        dec = self.drelu1(self.dbn1(self.deconv1(dec)))
        dec = self.drelu2(self.dbn2(self.deconv2(dec)))
        dec = self.drelu3(self.dbn3(self.deconv3(dec)))
        dec = self.drelu4(self.dbn4(self.deconv4(dec)))
        dec = self.deconv5(dec)  # Final layer, no activation to retain pixel values

        return dec


# Define the CNN Model using PyTorch Lightning
class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encode = MyEncoder()
        self.decode = MyDecoder()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x  # F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y_temp = batch
        y_hat = self(x)
        y = x * y_temp
        loss = torch.log(1 + F.mse_loss(y_hat, y))
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # psnr =
        self.log("train/loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_temp = batch
        y_hat = self(x)
        y = x * y_temp
        loss = torch.log(1 + F.mse_loss(y_hat, y))
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("validation/loss", loss, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)

        # Log images (only log a few samples)
        if batch_idx == 0:
            # grid_input = vutils.make_grid(y[:8], normalize=True, scale_each=True)
            # grid_pred = vutils.make_grid(y_hat[:8], normalize=True, scale_each=True)

            # self.logger.experiment.add_image("val/input_images", grid_input, self.global_step)
            # self.logger.experiment.add_image("val/predicted_images", grid_pred, self.global_step)

            # Ensure logger is WandbLogger and log images at batch_idx 0
            if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(y[:8], normalize=True, scale_each=True)
                grid_pred = vutils.make_grid(y_hat[:8], normalize=True, scale_each=True)

                self.logger.experiment.log(
                    {
                        "val/input_images": wandb.Image(
                            grid_input, caption="Input Images"
                        ),
                        "val/predicted_images": wandb.Image(
                            grid_pred, caption="Predicted Images"
                        ),
                        "global_step": self.trainer.global_step,
                    }
                )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)  # Learnable attention weights

    def forward(self, lstm_out):
        """
        lstm_out: [batch_size, sequence_length, hidden_size]
        Returns: Weighted hidden state [batch_size, hidden_size]
        """
        attn_weights = torch.softmax(
            self.attn(lstm_out), dim=1
        )  # Compute attention scores
        weighted_sum = torch.sum(
            attn_weights * lstm_out, dim=1
        )  # Apply attention weights
        return weighted_sum


class LSTMTimeSeriesClassifier(pl.LightningModule):
    def __init__(self, hidden_size=128, num_layers=2, lr=1e-3, use_attention=True):
        super().__init__()

        self.save_hyperparameters()  # Logs hyperparameters for WandB
        self.lr = lr
        # Define LSTM
        import torch.nn as nn

        lstm = nn.LSTM(input_size=11, hidden_size=128, num_layers=2, batch_first=True)

        segmodel = LitCNN.load_from_checkpoint(encoder_weight_path)
        encoder = segmodel.encode
        encoder.to(device="cpu")
        self.encoder = encoder

        # **Freeze the encoder weights**
        for param in self.encoder.parameters():
            param.requires_grad = False

        # **Freeze the encoder weights**
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_size, 1)

        self.use_attention = use_attention  # Toggle attention usage
        if use_attention:
            self.attention = Attention(hidden_size)

        # Loss Function
        self.loss_fn = nn.MSELoss()  # More stable than BCELoss
        self.accuracy_metric = BinaryAccuracy()  # Accuracy metric using TorchMetrics
        self.auc_metric = torchmetrics.AUROC(task="binary")

        self.preds_ = []
        self.targets_ = []

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

        # Extract features from encoder
        x1 = self.encoder(x1)  # Shape: [batch_size, 4, 4, 4]
        x2 = self.encoder(x2)

        # Reshape and stack to create time sequence
        x = torch.stack(
            [x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)], dim=1
        )  # Shape: [batch_size, 2, encoded_dim]

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output shape: [batch_size, 2, hidden_size]
        if self.use_attention:
            lstm_out = self.attention(lstm_out)  # Weighted hidden state
        else:
            lstm_out = lstm_out[:, -1, :]  # Default: Take last step

        logits = self.fc(lstm_out)  # Classification output
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

        self.preds_.append(probs.cpu())
        self.targets_.append(y.cpu())

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
        accuracy = self.accuracy_metric.compute()
        auc = self.auc_metric.compute()

        self.log("validation/accuracy", accuracy, prog_bar=True)
        self.log("validation/auc", auc, prog_bar=True)

        # # Concatenate stored predictions & targets
        # print(self.preds_)
        # print(self.targets_)
        preds_s = torch.cat(self.preds_, dim=0)
        targets_s = torch.cat(self.targets_, dim=0)

        # Log to WandB
        self.logger.log_table(
            key="validation/predictions_vs_ground_truth",
            columns=["Ground Truth", "Predictions"],
            data=list(zip(targets_s.cpu().numpy(), preds_s.cpu().numpy())),
        )

        # Reset stored predictions & targets
        self.preds_.clear()
        self.targets_.clear()

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4
        )  # L2 regularization
        return optimizer


class RadiomicsLSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size=11,
        hidden_size=22,
        num_layers=2,
        lr=1e-3,
        use_attention=True,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate  # Ensure learning rate is defined
        self.save_hyperparameters()  # Logs hyperparameters for WandB
        self.lr = lr
        # Define LSTM

        self.lstm = nn.LSTM(
            input_size=input_size,  # Number of radiomics features per MRI session
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

        self.use_attention = use_attention  # Toggle attention usage
        if use_attention:
            self.attention = Attention(hidden_size)

        # Loss Function
        self.loss_fn = nn.MSELoss()  # More stable than BCELoss
        self.accuracy_metric = BinaryAccuracy()  # Accuracy metric using TorchMetrics
        self.auc_metric = torchmetrics.AUROC(task="binary")

        self.preds_ = []
        self.targets_ = []

    def on_train_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_validation_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_test_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def _common_step(self, batch, batch_idx):
        x, y, _ = batch

        scores = self.forward(x)  # Forward pass through LSTM
        y = y.unsqueeze(1)  # Shape: [16, 1]

        loss = self.loss_fn(
            scores, y.float()
        )  # Ensure labels are float for BCEWithLogitsLoss
        print("common_step loss shape: ", loss.shape)
        print("common_step score shape: ", scores.shape)
        print("common_step y shape: ", y.shape)
        print("common_step x shape: ", x.shape)
        return loss, scores, y, x

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Input:
        - x: Tensor of shape (batch_size, sequence_length=2, input_dim)

        Output:
        - logits: Tensor of shape (batch_size, output_dim)
        """
        _, (h_n, _) = self.lstm(x)  # Get last hidden state
        logits = self.fc(h_n[-1])  # Use last layer's hidden state for classification
        return logits

    def training_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)  # Convert logits to probabilities
        print("loss shape: ", loss.shape)
        print("predictions shape: ", probs.shape)
        print("targets shape: ", y.shape)
        print("scores shape: ", scores.shape)
        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # Disable gradient computation

            loss, scores, y, x = self._common_step(batch, batch_idx)
            probs = torch.sigmoid(scores)

            # Ensure y has the shape [batch_size, 1]
            if y.dim() == 3:  # If y has shape [batch_size, 1, 1]
                y = y.squeeze(-1)  # Remove the last dimension

            self.preds_.append(probs.cpu())
            self.targets_.append(y.cpu())

            # Update metrics
            self.accuracy_metric.update(probs, y.int())
            self.auc_metric.update(probs, y.int())

            self.log(
                "validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)
        y = y.unsqueeze(1)
        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("train/auc", self.auc_metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        accuracy = self.accuracy_metric.compute()
        auc = self.auc_metric.compute()

        self.log("validation/accuracy", accuracy, prog_bar=True)
        self.log("validation/auc", auc, prog_bar=True)

        # # Concatenate stored predictions & targets
        # print(self.preds_)
        # print(self.targets_)
        preds_s = torch.cat(self.preds_, dim=0)
        targets_s = torch.cat(self.targets_, dim=0)

        # Log to WandB
        self.logger.log_table(
            key="validation/predictions_vs_ground_truth",
            columns=["Ground Truth", "Predictions"],
            data=list(zip(targets_s.cpu().numpy(), preds_s.cpu().numpy())),
        )

        # Reset stored predictions & targets
        self.preds_.clear()
        self.targets_.clear()

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
