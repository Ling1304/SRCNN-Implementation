# SRCNN Implementation 

This repository provides a comprehensive implementation of the paper **[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092)**. The paper introduces the concept of using deep learning-based Convolutional Neural Networks (CNNs) to enhance the resolution of low-resolution images efficiently.

This repository implements the SRCNN architecture using PyTorch and provides a clear and beginner-friendly introduction to deep learning and convolutional neural networks (CNNs)!

---

## 🛠️ How Does SRCNN Work?
<img src="pictures/diagram.png" alt="SRCNN Architecture" width="800"/>

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

### Comparison: Low-Resolution vs Super-Resolution (SRCNN Output)  

| **Low-Resolution Image**                         | **SRCNN Super-Resolution Image (x3)**                       | **Original High-Resolution Image**                        |
|--------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------|
| <img src="pictures/lr_butterfly.png" alt="Low Resolution" width="400"/> | <img src="pictures/lr_butterfly_x3.png" alt="Super Resolution" width="400"/> | <img src="pictures/original.png" alt="High Resolution" width="400"/> |
| PSNR: **23.03 dB** (LR vs HR)                     | PSNR: **23.13 dB** (SR vs HR)                                |                                                          |

---

## 📂 Repository Structure  
Below is the structure of the repository along with a brief description of each file:

### `src` Folder
- **`data_preprocessing.py`**  
  Script for pre-processing training images. It performs operations such as:  
  - Cropping sub-images of size 33x33.  
  - Blurring and downscaling images to generate low-resolution input for training.  

- **`utils.py`**  
  Contains utility functions such as:  
  - Converting images to tensors (and vice versa).  
  - Extracting the Y channel from images (and vice versa).  
  - PSNR (Peak Signal-to-Noise Ratio) calculation.  
  - Saving and loading the trained model.  

- **`dataset.py`**  
  Contains the custom dataset class used for handling low- and high-resolution image pairs. This also includes functionality to extract the Y channel from images for training. 

- **`model.py`**  
  Implements the SRCNN model, including the architecture with three convolutional layers and the weight initialization function.  

- **`train.py`**  
  Script for training the SRCNN model. It includes setting up the optimizer, defining the loss function, and saving the trained model checkpoints.  

- **`inference.py`**  
  Script for running inference with the trained model. It takes an input image and outputs the super-resolution image.  

### `data` Folder
- Contains a `train` folder with the **T91 training images** (zipped).  
- Contains a `validation` folder with the **Set14** and **Set9 validation images** (zipped).  

### `SRCNN model` Folder
- Contains the trained model file (`SRCNN_model_x2.pth`) with **2000 epochs**.

---

## 🏋️‍♂️ Training Process
<img src="pictures/training_validation_psnr.png" alt="SRCNN Architecture" width="800"/>

| **Metric**              | **Training**   | **Validation** |
|--------------------------|----------------|----------------|
| **Loss**                | 0.000108       | 0.0019       |
| **PSNR (dB)**           | 27.937         | 28.651        |

- **Dataset**:  
  - **Training Data**:  
    The 'T91' dataset, containing 91 images, was processed into two sets of 21,000 sub-images each. Each sub-image has a size of 33x33 with a stride of 14:  
    - **High-Resolution (HR) Set**: 21,000 center-cropped original sub-images.  
    - **Low-Resolution (LR) Set**: 21,000 low-resolution sub-images created by applying Gaussian blur and downscaling.  

  - **Validation Data**:  
    The 'Set5' and 'Set14' datasets were used for validation, consisting of two sets of 19 images each:  
    - **High-Resolution (HR) Set**: 19 center-cropped high-resolution images.  
    - **Low-Resolution (LR) Set**: 19 low-resolution images.  

  **Note**: Center cropping was applied due to no padding in the model, which results in smaller output images.
  
- **Model Layers (Patch Extraction and Representation )**:  
  - **Layer 1**:  
    - Input Channels: 1  
    - Output Channels: 64  
    - Kernel Size: 9x9  
    - Padding: 'valid' 

  - **Layer 2 (Non-linear Mapping)**:   
    - Input Channels: 64  
    - Output Channels: 32  
    - Kernel Size: 5x5  
    - Padding: 'valid' 

  - **Layer 3 (Reconstruction)**:  
    - Input Channels: 32  
    - Output Channels: 1  
    - Kernel Size: 5x5  
    - Padding: 'valid'  

- **Note**:  
  - ReLU activation is applied to **Layer 1** and **Layer 2**.  
  - No activation is applied to **Layer 3**.  
  - Padding is set to 'valid', meaning no padding is used to avoid border effects during training 



