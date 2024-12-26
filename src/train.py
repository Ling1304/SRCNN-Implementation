import time
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from model import SRCNN
from datasets import load_datasets
from utils import PSNR, save_model_state

# SOURCES:
# - https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
# - https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
# - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# - https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
# - https://pytorch.org/docs/stable/generated/torch.no_grad.html

def train(model, train_loader):
  """
  Function to train the CNN model with training data
  """
  # Set model to training mode
  model.train()

  # Set initial training loss and psnr to 0.0
  running_loss = 0.0
  running_psnr = 0.0

  # Loop through training dataset batch by batch
  for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    # Move the LR and HR training images to device (GPU or CPU)
    lr_image = data[0].to(device)
    hr_image = data[1].to(device)

    # Clear gradients stored previously, to prevent wrong weight updates
    optimizer.zero_grad()

    # Input LR image into model
    outputs = model(lr_image)
    # Calculate loss for current batch
    loss = loss_function(outputs, hr_image)
    # Backpropagation
    loss.backward()
    # Update the weights
    optimizer.step()

    # Add the total loss (to get total loss for current epoch)
    running_loss += loss.item()

    # Calculate PSNR for current batch
    batch_psnr = PSNR(hr_image, outputs)
    # Add the total PSNR (to get total PSNR for current epoch)
    running_psnr += batch_psnr

  # Calcualte the average loss and PSNR for current epoch
  avg_loss = running_loss/len(train_loader.dataset)
  avg_psnr = running_psnr/len(train_loader)

  return avg_loss, avg_psnr

def validate(model, val_loader, epoch):
  """
  Function to perform validation on CNN model with validation data
  """
  # Set model to evaluation mode (accept )
  model.eval()

  # Set initial validation loss and PSNR
  running_loss = 0.0
  running_psnr = 0.0

  # Disable gradient calculation (for inference/validation)
  with torch.no_grad():
    # Loop through each images in Set5
    for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
      # Move the LR and HR validation images to device (GPU or CPU)
      lr_image = data[0].to(device)
      hr_image = data[1].to(device)

      # Input LR image into model
      outputs = model(lr_image)
      # Calculate loss for current batch
      loss = loss_function(outputs, hr_image)

      # Add the total loss (to get total loss for current epoch)
      running_loss += loss.item()

      # Calculate PSNR for current batch
      batch_psnr = PSNR(hr_image, outputs)
      # Add the total PSNR (to get total PSNR for current epoch)
      running_psnr += batch_psnr

      if epoch % 500 == 0:
        save_image(outputs, f"C:/Users/Hezron Ling/Desktop/SRCNN_model/Validation Image/{epoch}.png")

  avg_loss = running_loss/len(val_loader.dataset)
  avg_psnr = running_psnr/len(val_loader)

  return avg_loss, avg_psnr

if __name__ == '__main__':
  # Training parameters
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Initialize the model
  print('Device: ', device)
  model = SRCNN().to(device)

  # Initialize the optimizer (SGD)
  optimizer = optim.SGD(
      [{"params": model.patch_extraction.parameters()},
      {"params": model.nonLinear_map.parameters()},
      {"params": model.reconstruction.parameters(), "lr": 1e-5}], # Set to 1e-5 for the last layer
      lr=1e-4,
      momentum=0.9,
      weight_decay=1e-4
  )
  # Initialize the loss function (used to update weights during training)
  loss_function = nn.MSELoss()

  # Load the training and validation data
  lr_train_path = 'C:/Users/Hezron Ling/Desktop/data_SRCNN_x3/Train/lr_sub_images'
  hr_train_path = 'C:/Users/Hezron Ling/Desktop/data_SRCNN_x3/Train/hr_sub_images_center'
  lr_val_path = 'C:/Users/Hezron Ling/Desktop/data_SRCNN_x3/Val/lr_image'
  hr_val_path = 'C:/Users/Hezron Ling/Desktop/data_SRCNN_x3/Val/hr_image_center'

  train_loader, val_loader = load_datasets(lr_train_path, hr_train_path, lr_val_path, hr_val_path)

  # Create a list to store running loss and PSNR
  train_loss, val_loss = [], []
  train_psnr, val_psnr = [], []

  # Set the epochs
  epochs = 2000

  # Start training and validation
  for epoch in range(epochs):
    print(f"Epoch: {epoch+1} of {epochs}")

    # Train the model and get the training loss and PSNR
    train_epoch_loss, train_epoch_psnr = train(model, train_loader)

    # Validate the model and get the validation loss and PSNR
    val_epoch_loss, val_epoch_psnr = validate(model, val_loader, epoch)

    # Store the running loss and PSNR
    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)

    # Print the train and validation PSNR every 50 epochs
    if (epoch+1) % 50 == 0:
      print(f"Train PSNR: {train_epoch_psnr:.3f}")
      print(f"Validation PSNR: {val_epoch_psnr:.3f}")

    # Save the model state dictionary every 100 epochs
    save_model_state(model, epoch, 'C:/Users/Hezron Ling/Desktop/SRCNN_model')

  print("Training done!")

