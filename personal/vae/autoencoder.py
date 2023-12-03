import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms

from vae.utils import ApplyPad

matplotlib.use('TkAgg')  # Use Tkinter as the backend
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Create an instance of the Encoder
IMAGE_SIZE = 32
CHANNELS = 1 
EMBEDDING_DIM = 2

# Transformations to apply to the images
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(custom_preprocess),
    ApplyPad(),
])



# Load Fashion MNIST dataset with the custom transform
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True , transform=custom_transform)

# Create the Data Loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

# The Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.shape_before_flattening = (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
        self.fc = nn.Linear(np.prod(self.shape_before_flattening), embedding_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        


        x = self.flatten(x)
        x = self.fc(x)

        return x

# Create an instance of the Encoder
encoder = Encoder(input_channels=CHANNELS, embedding_dim=EMBEDDING_DIM)


# The Decoder
class Decoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()
        self.shape_before_flattening = shape_before_flattening

        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))

        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            output_padding=1,  # To ensure stride works properly
        )
        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.conv_transpose3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=input_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = self.fc(x)
        # x = self.reshape(x)
        reshaping = (x.size(0), *self.shape_before_flattening)
        x = x.reshape(reshaping)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        return x

# Create an instance of the Decoder
decoder = Decoder(input_channels=CHANNELS, embedding_dim=EMBEDDING_DIM, shape_before_flattening=encoder.shape_before_flattening)

# The Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels, embedding_dim)
        self.decoder = Decoder(input_channels, embedding_dim, shape_before_flattening=encoder.shape_before_flattening)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Create an instance
autoencoder = Autoencoder(input_channels=CHANNELS, embedding_dim=EMBEDDING_DIM)


# --------------------------------- #
# --------------------------------- #  
#       Dry Run
# --------------------------------- # 
# --------------------------------- # 
# ---------------- #
# Example
# ---------------- #
iter_loader = iter(train_loader)
b1 = next(iter_loader)
b11 = b1[0]  # ([64, 1, 32, 32])
c11 = encoder(b11) # ([64, 2])
d11 = decoder(c11) # ([64, 1, 32, 32])
d12 = autoencoder(b11)  # ([64, 1, 32, 32])


sample1 = d12[0,0,:,:].detach().numpy()
plt.imshow(sample1, cmap="gray_r")
plt.show()

# ---------------- #
# Debugging DECODER
# ---------------- #
# Debug decoder
x = c11 = encoder(b11)
x1 = decoder.fc(x)
reshaping = (x1.size(0), *decoder.shape_before_flattening)
x1 = x1.reshape(reshaping)  # ([64, 128, 4, 4])
x2 = decoder.conv_transpose1(x1)  # ([64, 128, 8, 8])
x3 = decoder.conv_transpose2(x2)  # ([64, 64, 16, 16])


# --------------------------------- #
# --------------------------------- #  
#       Insights
# --------------------------------- # 
# --------------------------------- # 

####################
# ENCODER SUMMARY
####################
# Specify the input size (channels, height, width)
input_size = (1, 32, 32)

# Summary
summary(encoder, input_size=input_size)

####################
# DECODER SUMMARY
####################
# Specify the input size (channels, height, width)
input_size = (2,)

# Summary
summary(decoder, input_size=input_size)







# loader = iter(train_loader)
# b1 = next(loader)




