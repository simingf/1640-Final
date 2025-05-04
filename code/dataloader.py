import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision.transforms import functional as TF
from torchvision.transforms import ToTensor
from PIL import Image
import os

class RotationDataset(Dataset):
    def __init__(self, images, image_size, preprocess_func=None):
        self.images = images
        self.image_size = image_size
        self.preprocess_func = preprocess_func
        self.input_shape = self.images.shape[1:]
        assert(len(self.input_shape) == 3)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        rotation_angle = random.randint(0, 359)
        image = Image.fromarray(image.squeeze().astype(np.uint8))
        image = image.resize(self.image_size, Image.BILINEAR)
        rotated_image = TF.rotate(image, angle=rotation_angle)
        rotated_image = ToTensor()(rotated_image)
        if self.preprocess_func:
            rotated_image = self.preprocess_func(rotated_image)
        label = torch.tensor(rotation_angle, dtype=torch.long)
        one_hot_label = torch.zeros(360, dtype=torch.float)
        one_hot_label[(label - 2) % 360] = 0.05
        one_hot_label[(label - 1) % 360] = 0.05
        one_hot_label[label] = 0.8
        one_hot_label[(label + 1) % 360] = 0.05
        one_hot_label[(label + 2) % 360] = 0.05
        return rotated_image, one_hot_label

def get_dataloader(start_idx=0, num_images=10000, image_size=(640, 640), batch_size=64, shuffle=True, preprocess_func=None):
    image_dir = os.path.join(os.path.dirname(__file__), "../dataset")
    image_file_paths = [os.path.join(image_dir, f"{i}.png") for i in range(start_idx, start_idx + num_images)]
    images = np.array([np.array(Image.open(img_path)) for img_path in image_file_paths])
    dataset = RotationDataset(images, image_size=image_size, preprocess_func=preprocess_func)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_train_dataloader(num_images=8000, image_size=(640, 640), batch_size=64, shuffle=True, preprocess_func=None):
    return get_dataloader(0, num_images, image_size, batch_size, shuffle, preprocess_func)

def get_test_dataloader(num_images=2000, image_size=(640, 640), batch_size=64, shuffle=False, preprocess_func=None):
    return get_dataloader(8000, num_images, image_size, batch_size, shuffle, preprocess_func)

if __name__ == "__main__":
    dataloader = get_dataloader(num_images=128, image_size=(64, 64))

    for batch in dataloader:
        images, labels = batch
        one_hot = torch.nn.functional.one_hot(labels, num_classes=360).float()
        print(images.shape, labels.shape, one_hot.shape)