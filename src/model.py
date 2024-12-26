import math
import torch
from torch import nn

# SOURCES
# - https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
# - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# - https://pytorch.org/docs/stable/nn.init.html

class SRCNN(nn.Module): # Parent class 'nn.Module'
  def __init__(self):
    super().__init__() # Initialize parent class 'nn.Module'

    # Patch extraction and representation (First Layer)
    self.patch_extraction = nn.Conv2d(1, 64, kernel_size=(9,9), padding='valid')

    # Non-linear mapping (Second Layer)
    self.nonLinear_map = nn.Conv2d(64, 32, kernel_size=(1,1), padding='valid')

    # Reconstruction (Third Layer)
    self.reconstruction = nn.Conv2d(32, 1, kernel_size=(5,5), padding='valid')

    # ReLU activation
    self.relu = nn.ReLU()

    # Initialize weights
    self.initialize_weights()

  def forward(self, x):
    # Apply ReLU activation for the first 2 layers
    F1Y = self.relu(self.patch_extraction(x))
    F2Y = self.relu(self.nonLinear_map(F1Y))

    # No activation in last layer
    FY = self.reconstruction(F2Y)
    return FY

  # Weights are initialized with values drawn randomly from a Gaussian distribution (mean = 0, standard deviation = 0.001)
  # Biases are initialized to 0
  def initialize_weights(self):
    for module in self.modules(): # Loop through all modules
      if isinstance(module, nn.Conv2d): # Check if module is Conv2d layer
        nn.init.normal_(module.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(module.bias.data)

