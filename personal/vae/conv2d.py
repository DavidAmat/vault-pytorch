import torch
import torch.nn as nn

# Create a random 3x224x224 tensor
random_tensor = torch.rand(1, 3, 224, 224)

# Define the number of output filters (you can change this as needed)
num_filters = 16

# Define the convolutional layer
conv2d_layer = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1, padding=1)

# conv2d_layer.weight.shape
# torch.Size([16, 3, 3, 3])

# conv2d_layer.bias.shape
# torch.Size([16])

# Apply the convolutional layer to the random tensor
output_tensor = conv2d_layer(random_tensor)

# Print the shapes of the input and output tensors
print("Input Tensor Shape:", random_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)