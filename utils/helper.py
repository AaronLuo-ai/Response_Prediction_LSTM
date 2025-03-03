import cv2
import nrrd
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda

# class cross_entropy(self, input, target):
#     def __init__(self):
#         self.input = input
#         self.target = target
#
#     def forward(self, input, target):
#         return torch.mean(-torch.sum(target * torch.log(input), 1))


def clamp_values(image):
    return torch.clamp(image, 0, 1)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def resize_np_array(img, size=(224, 224)):
    return cv2.resize(img, size)


def grayscale_to_rgb(
    grayscale_tensor,
):  # called every time when the dataloader is called
    rgb_tensor = grayscale_tensor.repeat(3, 1, 1)
    return rgb_tensor


def change_dimension(nrrd_array):
    resized_array = []
    for i in range(nrrd_array.shape[0]):
        slice_2d = nrrd_array[i]  # Get the i-th 2D slice from the 3D array
        resized_slice = cv2.resize(slice_2d, (128, 128))  # Resize the 2D slice
        resized_slice = normalize(resized_slice)
        resized_array.append(resized_slice)
    output = np.stack(resized_array)
    return output
