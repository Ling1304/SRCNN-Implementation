import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from model import SRCNN

# Define inference function
def inference(image_path, model_path, output_path):
    """
    Function to perform inference using the trained SRCNN model with padding.
    
    Args:
        image_path (str): Path to the input low-resolution image.
        model_path (str): Path to the trained model .pth file.
        output_path (str): Path to save the super-resolved image.
    """
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load the trained model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Read the input image
    lr_image = cv2.imread(image_path)
    if lr_image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert BGR to YCrCb and extract the Y channel
    lr_image_ycrcb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)
    lr_image_y = lr_image_ycrcb[:, :, 0].astype(np.float32) / 255.0

    lr_image_y = np.expand_dims(lr_image_y, axis=0)
    lr_tensor = torch.tensor(lr_image_y, dtype=torch.float).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        sr_tensor = model(lr_tensor)  # Shape: [1, 1, H_out, W_out]

    # Convert tensor back to image (de-normalize)
    sr_image_y = sr_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255.0  
    sr_image_y = sr_image_y.clip(0, 255).astype('uint8')

    # Replace the Y channel in the original image with the super-resolution Y channel
    lr_image_ycrcb[:, :, 0] = sr_image_y
    sr_image = cv2.cvtColor(lr_image_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Save the super-resolved image
    cv2.imwrite(output_path, sr_image)
    print(f"Super-resolved image saved to {output_path}")

if __name__ == "__main__":
    image_path = "C:/Users/Hezron Ling/Desktop/data_SRCNN/Validation(5+14)/lr_image/lr_baby.png"  # Path to input image
    model_path = "C:/Users/Hezron Ling/Desktop/SRCNN_model/SRCNN_model_1000.pth"  # Path to trained model
    output_path = "C:/Users/Hezron Ling/Desktop/data_SRCNN/OUTPUT/output_sr_image.png"  # Path to save output image

    inference(image_path, model_path, output_path)
