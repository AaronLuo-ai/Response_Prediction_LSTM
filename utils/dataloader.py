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
import torchvision.transforms as T

train_transform = T.Compose(
    [
        T.CenterCrop((256, 256)),  # Crop to (256, 256)
        T.RandomRotation(degrees=15),  # Randomly rotate ±15 degrees
        T.Normalize(mean=[0.5], std=[0.5]),  # Normalize (replace with computed values)
    ]
)


test_transform = T.Compose(
    [
        T.CenterCrop((256, 256)),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class TwoPartDataset(Dataset):
    def __init__(
        self,
        root_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
        batch_path="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
        response_dir=r"C:\Users\aaron.l\Documents\db_20241213.xlsx",
        # This is the address of the data on my own computer (IGNORE THIS)
        # root_dir="/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple",
        # batch_path="/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple/batch.csv",
        # response_dir="/Users/luozisheng/Documents/Zhu_lab/db_20241213.xlsx",
        phase="train",
        transforms=None,
    ):
        self.slices = 0
        self.data_dir = Path(root_dir)
        self.batch_path = batch_path
        self.response_dir = response_dir
        self.phase = phase
        self.transforms = transforms

        df = pd.read_excel(response_dir)
        df["patient_info"] = list(
            zip(df["cnda_session_label"], df["AJCC Stage grouping "])
        )
        new_df = df[["patient_info"]]
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        new_df = new_df.drop_duplicates(subset=["patient_info"], keep="first")
        new_df = new_df[new_df["patient_info"].apply(lambda x: not pd.isna(x[1]))]
        new_df["patient_info"] = new_df["patient_info"].apply(
            lambda x: (x[0], 0) if x[1] in [0, 1] else (x[0], 1)
        )
        csv = pd.read_csv(batch_path)
        image_files = csv["Image"].tolist()
        mr1_patients = set(f.split("_MR1")[0] for f in image_files if "MR1" in f)
        mr2_patients = set(f.split("_MR2")[0] for f in image_files if "MR2" in f)
        patients_with_both = mr1_patients.intersection(mr2_patients)
        print("patients_with_both", sorted(patients_with_both))
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
            print("Train")
            print("response list: ", sorted(response_list))
            print("no response list: ", sorted(no_response_list))
            print("response list", len(response_list))
            print("no response list", len(no_response_list))
        else:
            response_list = response_list[int(len(response_list) * 0.75) :]
            no_response_list = no_response_list[int(len(no_response_list) * 0.75) :]
            combined_list = response_list + no_response_list
            print("Test")
            print("response list: ", sorted(response_list))
            print("no response list: ", sorted(no_response_list))
            print("response list", len(response_list))
            print("no response list", len(no_response_list))

        self.patient_data = []
        num_response = 0
        total_response = 0
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
                num_response += 1 if response == 1 else 0
                total_response += 1
        print(f"{num_response/total_response}%", response)

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        entry = self.patient_data[idx]

        mr1 = entry["MR1"]
        mr2 = entry["MR2"]
        response = entry["response"]  # Label (y)

        mr1 = torch.tensor(mr1, dtype=torch.float32).unsqueeze(0)
        mr2 = torch.tensor(mr2, dtype=torch.float32).unsqueeze(0)

        if self.transforms:
            mr1 = self.transforms(mr1)
            mr2 = self.transforms(mr2)

        return mr1, mr2, response


def main():
    data_dir = Path("/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple")
    batch_path = Path(
        "/Users/luozisheng/Documents/Zhu_lab/nrrd_images_masks_simple/batch.csv"
    )
    response_dir = Path("/Users/luozisheng/Documents/Zhu_lab/db_20241213.xlsx")
    phase = "test"

    dataset = TwoPartDataset(phase=phase, transforms=train_transform)
    Dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    print("len(Dataloader)", len(Dataloader))

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
