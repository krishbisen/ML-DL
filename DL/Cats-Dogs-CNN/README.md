# Cats vs Dogs Image Classification using CNN and PyTorch

A hands-on Deep Learning project for **binary image classification** using a Convolutional Neural Network (CNN) built from scratch with **PyTorch**.

The model classifies images into two categories:

- 🐱 Cats
- 🐶 Dogs

## Project Overview

This project covers the complete image-classification pipeline:

1. Dataset loading and exploration
2. Image preprocessing
3. Data augmentation
4. Data normalization
5. PyTorch `ImageFolder` dataset creation
6. DataLoader preparation
7. CNN architecture design
8. Model training and validation
9. Best-model selection
10. Test-set evaluation
11. Accuracy, precision, and recall
12. Confusion matrix analysis
13. Training/validation performance visualization

## Dataset

The project uses a Cats vs Dogs image classification dataset with the following structure:

```text
dogs-vs-cats-classification/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Dataset split used in the project:

| Split | Images |
|---|---:|
| Training | 19,943 |
| Validation | 2,492 |
| Test | 2,495 |
| **Total** | **24,930** |

Classes are mapped as:

```text
cats → 0
dogs → 1
```

## Preprocessing

Images are resized to **128 × 128** pixels.

### Training augmentation

- Resize to 128 × 128
- Random horizontal flip
- Random rotation up to 10°
- Convert image to tensor
- Normalize using ImageNet mean and standard deviation

### Validation/Test preprocessing

- Resize to 128 × 128
- Convert to tensor
- Normalize using the same mean and standard deviation

## CNN Architecture

The CNN was designed from scratch using PyTorch.

```text
Input Image
    ↓
Conv2D: 3 → 32
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D: 32 → 64
    ↓
ReLU
    ↓
MaxPooling
    ↓
Conv2D: 64 → 128
    ↓
ReLU
    ↓
MaxPooling
    ↓
Flatten
    ↓
Fully Connected: 128×16×16 → 256
    ↓
ReLU
    ↓
Dropout (0.5)
    ↓
Output: 1 neuron
```

Since this is a binary classification problem, the model uses a single output neuron with **BCEWithLogitsLoss**.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Image Size | 128 × 128 |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | BCEWithLogitsLoss |
| Dropout | 0.5 |
| Hardware | Kaggle GPU |

The model with the highest validation accuracy was saved as:

```text
best_cnn_model.pth
```

## Results

The trained CNN achieved the following performance on the test dataset:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **88.16%** |
| Test Accuracy | **88.22%** |
| Precision | **89.28%** |
| Recall | **86.85%** |

### Confusion Matrix

```text
                Predicted
              Cat      Dog
Actual Cat    1118     130
       Dog     164    1083
```

Interpretation:

- **1118 cats** were correctly classified as cats.
- **1083 dogs** were correctly classified as dogs.
- **130 cats** were classified as dogs.
- **164 dogs** were classified as cats.

## Visualizations

The notebook includes:

- Training vs validation accuracy curve
- Training vs validation loss curve
- Confusion matrix
- Model performance chart
- Sample predictions with actual and predicted labels

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebook
- NVIDIA GPU

## Project File

Main notebook:

```text
cats_dogs_cnn_pytorch.ipynb
```

The notebook contains the complete implementation from dataset loading through final evaluation.

## What I Learned

This project helped me understand the complete CNN workflow rather than only using a pre-trained model:

- How image data is loaded using `ImageFolder`
- Why preprocessing and normalization are required
- How data augmentation helps improve generalization
- How convolution layers extract image features
- The role of ReLU and max pooling
- How feature maps are converted into classification outputs
- How loss is calculated
- How backpropagation updates model weights
- How validation data is used to select the best model
- How accuracy, precision, recall, and confusion matrices evaluate a classifier

## Future Improvements

Possible next steps for this project:

- Transfer learning with ResNet or EfficientNet
- More advanced data augmentation
- Learning-rate scheduling
- Hyperparameter tuning
- Grad-CAM for visual model interpretation
- Real-time webcam classification
- Model deployment as a web or desktop application

## Author

**Krish Bisen**

B.Tech — Automation & Robotics  
Dr. D. Y. Patil Institute of Technology, Pimpri, Pune
