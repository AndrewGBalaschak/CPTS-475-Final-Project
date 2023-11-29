# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim import Adam
import torchvision.datasets as dsets

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import LeastSquaresGeneratorLoss, LeastSquaresDiscriminatorLoss
from torchgan.trainer import Trainer
from torchvision.transforms import transforms

# Define your custom dataset class
class NFUQNIDS(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.data = pd.read_csv(data_file)
        self.labels = self.data['Label']
        self.data = self.data.drop({'Label', 'Attack', 'Dataset'}, axis=1)
        self.data = self.data.values

        # Standardize the features
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels.iloc[idx]

        if self.transform:
            item = self.transform(item)

        if self.target_transform:
            label = self.target_transform(label)

        # Return data in the format expected by TorchGAN
        return {'data': torch.tensor(item, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)}

# Replace these with your actual values
data_file = 'your_data.csv'
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transform for your dataset (modify as needed)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Instantiate your custom dataset
nfu_dataset = NFUQNIDS(data_file=data_file, transform=transform)

# Define the network architecture and losses
dcgan_network = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "encoding_dims": 100,
            "out_channels": 1,
            "step_channels": 32,
            "nonlinearity": torch.nn.LeakyReLU(0.2),
            "last_nonlinearity": torch.nn.Tanh(),
        },
        "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_channels": 1,
            "step_channels": 32,
            "nonlinearity": torch.nn.LeakyReLU(0.2),
            "last_nonlinearity": torch.nn.LeakyReLU(0.2),
        },
        "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
    },
}
# Loss functions (you can choose the appropriate GAN loss for your task)
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

# Training parameters
# Training parameters
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    epochs = 10
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

# Trainer initialization and training
trainer = Trainer(dcgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device)
trainer(nfu_dataset)  # Assuming you use your custom dataset

# Visualizing the samples
# Grab a batch of real images from the dataloader
real_batch = next(iter(nfu_dataset))

# Plot the real images
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch['data'].to(device)[:64], padding=5, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, trainer.epochs)))
plt.show()