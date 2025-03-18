import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import openpyxl

features_file = (
    "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\radiomics_features.xlsx"
)
db_file = "C:\\Users\\aaron.l\\Documents\\db_20241213.xlsx"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# Load extracted radiomics features
df = pd.read_excel(features_file)

# Load patient response data
db = pd.read_excel(db_file, usecols=["cnda_subject_label", "AJCC Stage grouping "])
db = db.dropna(subset=["AJCC Stage grouping "])  # Remove patients with null responses
db["Response"] = db["AJCC Stage grouping "].apply(lambda x: 0 if x in {0, 1, 2} else 1)
db = db.drop_duplicates(subset=["cnda_subject_label"])  # Drop duplicate patients

# Normalize cnda_subject_label to match Patient_ID format
db["Patient_ID"] = db["cnda_subject_label"].str.replace("cnda_", "", regex=False)

# Merge response data with extracted features
df = df.merge(db[["Patient_ID", "Response"]], on="Patient_ID")

# Reshape dataset to have MR1 and MR2 features for each patient
# Ensure each patient has both MR1 and MR2 sessions
grouped = df.groupby("Patient_ID").filter(lambda x: set(x["Session"]) == {"MR1", "MR2"})
column_list = grouped.columns.tolist()
print("Column headers:", column_list)
print("Number of columns:", len(column_list))
# print("grouped: ",grouped)


def get_patient_features(patient_id, session):
    feature_columns = [
        col
        for col in grouped.columns
        if col not in {"Patient_ID", "Session", "Response"}
    ]
    return grouped.loc[
        (grouped["Patient_ID"] == patient_id) & (grouped["Session"] == session),
        feature_columns,
    ].values.flatten()


class RadiomicsDataset(Dataset):
    def __init__(self, dataframe):
        self.patients = dataframe["Patient_ID"].unique()
        self.dataframe = dataframe
        print("dataframe shape:", dataframe.shape)
        print("self.dataframe.columns.tolist(): ", self.dataframe.columns.tolist())
        print("list length: ", len(self.dataframe.columns.tolist()))
        self.labels = {
            row["Patient_ID"]: row["Response"]
            for _, row in dataframe.drop_duplicates("Patient_ID").iterrows()
        }

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        x_mr1 = torch.tensor(
            get_patient_features(patient_id, "MR1"), dtype=torch.float32
        )
        x_mr2 = torch.tensor(
            get_patient_features(patient_id, "MR2"), dtype=torch.float32
        )
        y = torch.tensor(self.labels[patient_id], dtype=torch.long)

        # Stack MR1 and MR2 along the sequence dimension -> (sequence_length=2, input_dim)
        x_sequence = torch.stack([x_mr1, x_mr2], dim=0)  # Shape: (2, num_features)

        return x_sequence, y, patient_id


def main():
    # Train-Test Split with Proportional Class Distribution
    train_df, test_df = train_test_split(
        grouped, test_size=0.25, stratify=grouped["Response"], random_state=42
    )

    # Create dataset instances
    train_dataset = RadiomicsDataset(train_df)
    test_dataset = RadiomicsDataset(test_df)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Iterating through the entire testing dataloader:")
    for batch_idx, (x_sequence, y, patient_ids) in enumerate(train_loader):
        if batch_idx == 0:
            print(x_sequence, y, patient_ids)
        print(f"Batch {batch_idx + 1}")
        for i in range(len(patient_ids)):
            print(f"Patient ID: {patient_ids[i]}, Label: {y[i].item()}")
        print()
        print("x_sequence type: ", type(x_sequence))
        print("x_sequence shape: ", x_sequence.shape)
        print("patient_ids: ", patient_ids)
        print("patient_ids type: ", type(patient_ids))

    print("Dataset preparation complete. Train and test datasets created.")


if __name__ == "__main__":
    main()
