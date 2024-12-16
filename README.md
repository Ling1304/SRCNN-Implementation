# SRCNN Implementation 

This repository provides a comprehensive implementation of the paper **[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092)**, authored by **Chao Dong**, **Chen Change Loy**, **Kaiming He**, and **Xiaoou Tang**. The paper introduces the concept of using deep learning-based Convolutional Neural Networks (CNNs) to enhance the resolution of low-resolution images efficiently.

This repository implements the SRCNN architecture using PyTorch and provides a clear and beginner-friendly introduction to deep learning and convolutional neural networks (CNNs)!

## 🔜 Project Status

This project is currently ongoing. Suggestions, and feedback are welcome. Stay tuned for more updates!

---

## 🛠️ How Does SRCNN Work?
![SRCNN Architecture](srcnn_diagram/diagram.png)

SRCNN is a simple but effective model consisting of three main layers:
1. **Patch Extraction Layer**: 
- Extract patches from a low resolution images based on the size of filters
- The filter then extracts an n1-dimensional feature for each patch
- The more filters a layer has, the more features it can capture, meaning the upscaled images are of higher quality
- Each filter has a certain size, larger filters can capture more image details

2. **Non-Linear Mapping Layer**: 
- This layer maps each of these n1-dimensional vectors into an n2-dimensional feature
- This mapping helps capture more complex patterns in the image by applying a non-linear mapping
- The non linear mapping is not on a patch of input image, but on a patch of the feature map
- This means the filter goes through each patch of the feature map (n1-dimensional)

3. **Reconstruction**: 
- Finally aggregate the high-resolution patch-wise representations to generate final high resolution image

---

## 📂 Repository Structure  
Below is the structure of the repository along with a brief description of each file:

- **`model.py`**:  
  Implements the SRCNN model, including the architecture with three convolutional layers and the weight initialization function.  

- **`dataset.py`**:  
  Contains the custom dataset class used for handling low- and high-resolution image pairs. This also includes functionality to extract the Y channel from images for training.  

- **`train.py`**:  
  Script for training the SRCNN model. It includes setting up the optimizer, defining the loss function, and saving the trained model checkpoints.  

- **`inference.py`**:  
  Script for running inference with the trained model. It takes an input image and outputs the super-resolution image.  

- **`utils.py`**:  
  Contains utility functions such as:  
    - Converting images to tensors (and vice versa).  
    - Extracting the Y channel from images (and vice versa).  
    - PSNR (Peak Signal-to-Noise Ratio) calculation.  
    - Saving and loading the trained model.  

- **`data_preprocessing.py`**:  
  Script for pre-processing training images. It performs operations such as:  
    - Cropping sub-images of size 33x33.  
    - Blurring and downscaling images to generate low-resolution input for training.  

---

## 🏋️‍♂️ Training Process

- **Dataset**:  
  - **Train**:  
    The 'T91' dataset, which consists of 91 images, was cropped into 2 sets of 21,000 sub-images with a sub-image size of 33 and a stride of 14:  
    - **Set 1 (high resolution)**: 21,000 original sub-images  
    - **Set 2 (low resolution)**: 21,000 low-resolution sub-images (created by applying Gaussian blur and rescaling)  

  - **Validation**:  
    The 'Set5' and 'Set14' datasets, which contain a total of 19 images:  
    - 19 low-resolution images  
    - 19 corresponding high-resolution images  
    - **Note**: These images were not cropped for training  

- **Model Layers**:  
  - **Layer 1**: 
  - **Layer 2**: 
  - **Layer 3**: 

  - **Note**: Padding is applied to ensure that the input and output images have the same size.  


