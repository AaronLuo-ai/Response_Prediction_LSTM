import pandas as pd
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import nrrd
import matplotlib.pyplot as plt
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Compose


train_transform_reg = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.CenterCrop(256, 256),
        A.Normalize(mean=[156.22], std=[132.88]),
        ToTensorV2(),
    ]
)

val_transform_reg = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256),
        A.Resize(256, 256),
        A.Normalize(mean=[156.22], std=[132.88]),
        ToTensorV2(),
    ]
)


class TwoPartDataset(Dataset):
    def __init__(
        self,
        # root_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
        # batch_path="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
        # response_dir=r"C:\Users\aaron.l\Documents\db_20241213.xlsx",
        # This is the address of the data on my own computer (IGNORE THIS)
        root_dir="/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple",
        batch_path="/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple/batch.csv",
        response_dir="/Users/luozisheng/Documents/Zhu_lab/db_20241213.xlsx",
        phase="train",
        transforms=None,
    ):
        self.slices = 0
        self.data_dir = Path(root_dir)
        self.batch_path = batch_path
        self.response_dir = response_dir
        self.phase = phase
        self.transforms = transforms
        self.num_slices = 0

        df = pd.read_excel(response_dir)
        df["patient_info"] = list(zip(df["cnda_session_label"], df["Tumor Response"]))
        new_df = df[["patient_info"]]
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        new_df = new_df.drop_duplicates(subset=["patient_info"], keep="first")
        new_df["patient_info"] = new_df["patient_info"].apply(
            lambda x: (x[0], 1) if x[1] == "Complete response" else (x[0], 0)
        )
        new_df = new_df.drop_duplicates(subset=["patient_info"], keep="first")
        csv = pd.read_csv(batch_path)
        image_files = csv["Image"].tolist()
        mr1_patients = set(f.split("_MR1")[0] for f in image_files if "MR1" in f)
        mr2_patients = set(f.split("_MR2")[0] for f in image_files if "MR2" in f)
        patients_with_both = mr1_patients.intersection(mr2_patients)
        response_map = {
            "_".join(x[0].split("_")[:2]): x[1]
            for x in new_df["patient_info"]
            if "_".join(x[0].split("_")[:2]) in patients_with_both
        }
        response_list = sorted(
            [patient for patient, response in response_map.items() if response == 1]
        )
        no_response_list = sorted(
            [patient for patient, response in response_map.items() if response == 0]
        )
        if phase == "train":
            response_list = response_list[: int(len(response_list) * 0.75)]
            no_response_list = no_response_list[: int(len(no_response_list) * 0.75)]
            combined_list = response_list + no_response_list
        else:
            response_list = response_list[: int(len(response_list) * 0.75) + 1]
            no_response_list = no_response_list[: int(len(no_response_list) * 0.75) + 1]
            combined_list = response_list + no_response_list

        self.patient_data = []
        patients = set(f.split("_MR")[0] for f in combined_list)
        for patient in sorted(patients):
            mr1_path = self.data_dir / Path(f"{patient}_MR1_images.nrrd")
            mr2_path = self.data_dir / Path(f"{patient}_MR2_images.nrrd")

            mr1_array, _ = nrrd.read(mr1_path)
            mr2_array, _ = nrrd.read(mr2_path)
            min_slices = min(mr1_array.shape[0], mr2_array.shape[0])

            mr1_array = mr1_array[:min_slices]
            mr2_array = mr2_array[:min_slices]
            response = response_map.get(patient, "Unknown")

            if response == "Unknown":
                print("There is unknown")
                continue
            for slice in range(min_slices):
                self.patient_data.append(
                    {
                        "MR1": mr1_array[slice],
                        "MR2": mr2_array[slice],
                        "response": response,
                    }
                )
            self.num_slices += min_slices

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        entry = self.patient_data[idx]

        mr1 = entry["MR1"]
        mr2 = entry["MR2"]
        response = entry["response"]  # Label (y)

        mr1 = torch.tensor(mr1, dtype=torch.float32)
        mr2 = torch.tensor(mr2, dtype=torch.float32)

        if self.transforms:
            mr1 = self.transforms(image=mr1.numpy())["image"]
            mr2 = self.transforms(image=mr2.numpy())["image"]
        return mr1, mr2, response


def main():
    data_dir = Path("/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple")
    batch_path = Path(
        "/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple/batch.csv"
    )
    response_dir = Path("/Users/luozisheng/Documents/Zhu_lab/db_20241213.xlsx")
    phase = "train"

    dataset = TwoPartDataset(phase=phase, transforms=train_transform_reg)
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

    for mr1_batch, mr2_batch, response_batch in Dataloader:
        batch_size = mr1_batch.shape[0]  # Number of images in batch
        fig, axes = plt.subplots(batch_size, 2, figsize=(6, 3 * batch_size))

        for i in range(batch_size):
            ax1, ax2 = axes[i] if batch_size > 1 else (axes[0], axes[1])
            ax1.imshow(mr1_batch[i].numpy().squeeze(), cmap="gray")
            ax1.set_title(f"MR1 - Response: {response_batch[i].item()}")
            ax1.axis("off")

            ax2.imshow(mr2_batch[i].numpy().squeeze(), cmap="gray")
            ax2.set_title(f"MR2 - Response: {response_batch[i].item()}")
            ax2.axis("off")

        plt.tight_layout()
        plt.show()
        break


if __name__ == "__main__":
    main()
