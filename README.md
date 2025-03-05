
---

## 📊 Overview

### Objective

The project focuses on predicting tumor responses based on MRI scans before and after treatment. Specifically, it aims to predict whether a tumor will have a complete response to treatment using data from `MR1` (pre-treatment MRI) and `MR2` (post-treatment MRI). The model used is an **LSTM**, which processes these MR1 and MR2 images as sequential data.

---

### Input Data

The data consists of:

- **MRI images**: Stored in NRRD format (`MR1` for pre-treatment, `MR2` for post-treatment).
- **Response labels**: Binary classification indicating the tumor's response to treatment ("Complete response" or "No response"). These labels are provided in an Excel file (`db_20241213.xlsx`).
- **Batch information**: Contains a list of patient data including the MRI filenames, stored in a CSV file (`batch.csv`).

---

### Model

The model consists of an **LSTM** network:

1. **Input layer**: Takes in a sequence of MRI images (pre-treatment and post-treatment).
2. **LSTM layers**: Processes the images sequentially to capture temporal patterns.
3. **Fully connected layer**: Output layer for binary classification, predicting tumor response.

---

### Training

The model is trained using:

- **Loss function**: Binary Cross-Entropy Loss (`BCEWithLogitsLoss`).
- **Optimizer**: Adam optimizer.
- **Metrics**: Accuracy and AUROC (Area Under the Receiver Operating Characteristic curve).

---

### Logging

We use **WandB** to log the training/validation loss and accuracy metrics during training.

---

## 🚀 Installation

To set up this project, clone the repository and install the required dependencies:

```bash
git clone <repo_url>
cd <repo_directory>
pip install -r requirements.txt
