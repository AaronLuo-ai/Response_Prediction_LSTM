import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, adjust_brightness


class JointTransformTrain:
    def __init__(
        self,
        degrees,
        transform_image,
        transform_mask,
        interpolation=InterpolationMode.NEAREST,
    ):
        self.degrees = degrees
        self.interpolation = interpolation
        # Define the other transforms
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.flip_prob = 0.5

    def __call__(self, image, mask):
        # Generate a single random rotation angle for both image and mask
        angle = random.uniform(-self.degrees, self.degrees)
        transformed_image = self.transform_image(image)
        transformed_mask = self.transform_mask(mask)

        rotated_image = F.rotate(
            transformed_image, angle, interpolation=self.interpolation
        )
        rotated_mask = F.rotate(
            transformed_mask, angle, interpolation=self.interpolation
        )

        if random.random() < self.flip_prob:
            rotated_image = F.hflip(rotated_image)
            rotated_mask = F.hflip(rotated_mask)

        brightness_factor = random.random() / 2.5 + 0.8  # scales between [0.8, 1.2)
        brightened_image = adjust_brightness(
            rotated_image, brightness_factor=brightness_factor
        )
        brightened_mask = rotated_mask
        return brightened_image, brightened_mask


class JointTransformTest:
    def __init__(self, transform_image, transform_mask):
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __call__(self, image, mask):
        # Apply the separate transforms to the image and mask
        transformed_image = self.transform_image(image)
        transformed_mask = self.transform_mask(mask)
        return transformed_image, transformed_mask
