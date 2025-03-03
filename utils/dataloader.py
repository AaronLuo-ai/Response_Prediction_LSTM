import pandas as pd
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import nrrd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from utils.augmentation import *
from helper import *


class SegmentImageDataset(Dataset):
    def __init__(
            self,
            data_dir="./data/nrrd_images_masks_simple",
            batch_path="./data/nrrd_images_masks_simple/batch.csv",
            phase="train",
            transform=None,
            target_transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.batch_path = Path(batch_path)
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        # Load image filenames from CSV
        csv = pd.read_csv(self.batch_path)
        num_lines = len(csv)
        separation_index = int(0.75 * num_lines)

        if self.phase == "train":
            self.image_files = csv["Image"].tolist()[:separation_index]
        else:
            self.image_files = csv["Image"].tolist()[separation_index + 1:]

        # List to store (image_path, mask_path, num_slices)
        self.data = []
        self.slice_indices = []  # Maps global index -> (file_idx, slice_idx)

        total_slices = 0
        for file_idx, image_file in enumerate(self.image_files):
            image_path = self.data_dir / image_file
            mask_file = image_file.replace("images", "masks")  # Find corresponding mask
            mask_path = self.data_dir / mask_file

            if not mask_path.exists():
                print(f"Warning: Mask {mask_path} not found for image {image_file}")
                continue  # Skip if mask doesn't exist

            # Read only the header to get the number of slices
            # print(image_path)
            header = nrrd.read_header(str(image_path))
            num_slices = header["sizes"][0]  # First dimension is slices

            self.data.append((image_path, mask_path, num_slices))

            # Create slice index mapping (global index → (file_idx, slice_idx))
            for slice_idx in range(num_slices):
                self.slice_indices.append((file_idx, slice_idx))

            total_slices += num_slices

        print(f"Total slices in dataset: {total_slices}")

    def __getitem__(self, index):
        file_idx, slice_idx = self.slice_indices[index]
        image_path, mask_path, _ = self.data[file_idx]

        # Load only the required slice
        image, _ = nrrd.read(str(image_path))
        mask, _ = nrrd.read(str(mask_path))

        # slice_image = Image.fromarray(image[slice_idx, :, :].astype("float32"))#.unsqueeze(0)
        # slice_mask = Image.fromarray(mask[slice_idx, :, :].astype("float32"))#.unsqueeze(0)

        slice_image = image[slice_idx, :, :].astype("float32")  # .unsqueeze(0)
        slice_mask = mask[slice_idx, :, :].astype("float32")  # .unsqueeze(0)

        # slice_image = torch.from_numpy(slice_image.astype("float32")).unsqueeze(0)  # Shape: (1, H, W)
        # slice_mask = torch.from_numpy(slice_mask.astype("float32")).unsqueeze(0)  # Shape: (1, H, W)
        if self.phase == 'train':
            transformed = train_transform(image=slice_image, mask=slice_mask)
            slice_image = Image.fromarray(transformed['image'])
            slice_mask = Image.fromarray(transformed['mask'])
        else:
            transformed = val_transform(image=slice_image, mask=slice_mask)
            slice_image = Image.fromarray(transformed['image'])
            slice_mask = Image.fromarray(transformed['mask'])

        if self.transform:
            slice_image = self.transform(slice_image)
        if self.target_transform:
            slice_mask = self.target_transform(slice_mask)

        return slice_image, slice_mask

    def __len__(self):
        return len(self.slice_indices)

def main():

    transform_train_mask = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_train_image = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_test_image = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_test_mask = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_train = JointTransformTrain(
        60, transform_image=transform_train_image, transform_mask=transform_train_mask
    )
    transform_test = JointTransformTest(transform_test_image, transform_test_mask)

    # dataset_train = MRIDataset(root_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
    #                            batch_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
    #                            phase="train", transform=transform_train)
    #
    # dataset_val = MRIDataset(root_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
    #                          batch_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
    #                          phase="val", transform=transform_test)

    # print("\n🔹 Training Set Patients:", len(dataset_train))  # ✅ Counts unique patients
    # print("🔹 Validation Set Patients:", len(dataset_val))  # ✅ Counts unique patients
    #
    # # ✅ Fetch a sample patient data
    # sample_images, sample_masks = dataset_train[0]
    #
    # print("\n🔹 Sample Image Tensor Shape:", sample_images.shape)  # Expecting [2, H, W]
    # print("🔹 Sample Mask Tensor Shape:", sample_masks.shape)  # Expecting [2, H, W]
    #
    # # ✅ Print All Patient IDs
    # print("\n🔹 TRAINING SET: Patients")
    # for patient in dataset_train.patients:
    #     print(f"Patient: {patient}")
    #
    # print("\n🔹 VALIDATION SET: Patients")
    # for patient in dataset_val.patients:
    #     print(f"Patient: {patient}")

if __name__ == '__main__':
        main()
