import torch
import h5py
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

def preprocess_mnist(train_save_path='mnist_rotated_train.h5', val_save_path='mnist_rotated_val.h5', digits=(1, 2)):
    """
    Load MNIST, filter digits, rotate images, and save in HDF5 format for both train and validation datasets.
    """
    angles = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
    transform = transforms.Compose([transforms.ToTensor()])

    # Process training dataset
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_images, train_labels = [], []
    for img, label in mnist_train:
        if label in digits:
            for angle in angles:
                rotated_img = img.squeeze().numpy()
                rotated_img = Image.fromarray((rotated_img * 255).astype(np.uint8)).rotate(angle)
                rotated_img = np.array(rotated_img, dtype=np.float32) / 255.0  # Normalize
                train_images.append(rotated_img)
                train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Save training dataset to HDF5
    with h5py.File(train_save_path, 'w') as hf:
        hf.create_dataset('images', data=train_images)
        hf.create_dataset('labels', data=train_labels)
    print(f"Training dataset saved to {train_save_path} with {len(train_images)} samples.")

    # Process validation dataset
    mnist_val = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_images, val_labels = [], []
    for img, label in mnist_val:
        if label in digits:
            for angle in angles:
                rotated_img = img.squeeze().numpy()
                rotated_img = Image.fromarray((rotated_img * 255).astype(np.uint8)).rotate(angle)
                rotated_img = np.array(rotated_img, dtype=np.float32) / 255.0  # Normalize
                val_images.append(rotated_img)
                val_labels.append(label)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    # Save validation dataset to HDF5
    with h5py.File(val_save_path, 'w') as hf:
        hf.create_dataset('images', data=val_images)
        hf.create_dataset('labels', data=val_labels)
    print(f"Validation dataset saved to {val_save_path} with {len(val_images)} samples.")

class RotatedMNIST(Dataset):
    """
    PyTorch Dataset class to load the preprocessed MNIST dataset.
    """
    def __init__(self, file_path='mnist_rotated.h5'):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as hf:
            self.data_len = len(hf['images'])
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hf:
            image = torch.tensor(hf['images'][idx]).unsqueeze(0)  # Add channel dimension
            label = torch.tensor(hf['labels'][idx], dtype=torch.long)
        return image, label

if __name__ == "__main__":
    preprocess_mnist()
    # Example usage
    dataset = RotatedMNIST(file_path='mnist_rotated_train.h5')
    print(f"Dataset size: {len(dataset)}")
    idx = np.random.randint(0, len(dataset))
    img, label = dataset[idx]
    print(f"Image shape: {img.shape}, Label: {label}")
    # Display the image (optional)
    img = img.squeeze().numpy()
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.show()
