import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision.transforms import functional as TF
from torchvision.transforms import ToTensor
from PIL import Image

class RotNetDataset(Dataset):
    def __init__(self, images, preprocess_func=None):
        """
        Args:
            images (numpy.ndarray): Array of images.
            preprocess_func (callable, optional): Optional preprocessing function.
        """
        self.images = images
        self.preprocess_func = preprocess_func
        self.input_shape = self.images.shape[1:]
        if len(self.input_shape) == 2:  # if grayscale
            self.input_shape = self.input_shape + (1,)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        rotation_angle = random.randint(0, 359)

        # Convert to PIL image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        image = Image.fromarray(image.squeeze().astype(np.uint8))

        # Rotate image
        rotated_image = TF.rotate(image, angle=rotation_angle)

        # Convert back to tensor and normalize to [0, 1]
        rotated_image = ToTensor()(rotated_image)  # shape: [C, H, W]

        # Apply any optional preprocessing
        if self.preprocess_func:
            rotated_image = self.preprocess_func(rotated_image)

        # Label: rotation angle as class index (0 to 359)
        label = torch.tensor(rotation_angle, dtype=torch.long)

        return rotated_image, label

if __name__ == "__main__":
    # Example usage
    # Create a random dataset of images
    num_images = 1000
    height, width = 64, 64
    images = np.random.randint(0, 256, (num_images, height, width), dtype=np.uint8)

    # Create dataset and dataloader
    dataset = RotNetDataset(images, preprocess_func=None)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the dataloader
    for batch in dataloader:
        images, labels = batch
        one_hot = torch.nn.functional.one_hot(labels, num_classes=360).float()
        print(images.shape, labels.shape)
        break