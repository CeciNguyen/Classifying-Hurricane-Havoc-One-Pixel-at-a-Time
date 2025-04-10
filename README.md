# Classifying Hurricane Havoc One Pixel at a Time
This project applies deep learning techniques to assess building damage using satellite imagery collected in the aftermath of Hurricane Harvey. Leveraging Convolutional Neural Networks (CNNs), we aim to accurately classify buildings as either damaged or undamaged, contributing to more efficient disaster response and recovery efforts.

## Data Preparation
- Images were cropped and resized to 128×128 pixels.
- Only grayscale (single-channel) images were used to reduce complexity.
- Labels were simplified to two classes:
    - 0: No damage
    - 1: Damaged
- The data was split into training and test sets using an 80/20 ratio.
- Preprocessing included normalization to a [0, 1] range.
  
## Methodology

Three CNN models were implemented and evaluated:

1. **Baseline CNN**  
   A custom architecture with three convolutional layers followed by fully connected layers. This model achieved strong performance despite its simplicity.

2. **LeNet-5**  
   The classic LeNet-5 architecture, accepted 128x128 input images and used larger kernel sizes and padding to accommodate the larger input size.

3. **Alternate-LeNet-5**  
   An adaptation of the classic LeNet-5 architecture, modified to accept 128x128 input images and tailored for satellite data rather than handwritten digits.

## Performance Summary

| Model                | Test Accuracy |
|----------------------|---------------|
| Baseline CNN         | ~96%          |
| LeNet-5              | ~92%          |
| Alternate-LeNet-5    | ~97%          |

Notably, the baseline CNN outperformed the LeNet-based model, likely due to its more flexible structure and suitability for the given image size and content. However, the Alternate LeNet-5 took the cake overall!

## Model Deployment 
This section can be found under the hurrican_model_deployment folder of the repository.
