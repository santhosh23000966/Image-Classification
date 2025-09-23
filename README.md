# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Dataset
The FashionMNIST dataset consists of 70,000 grayscale images of size 28 × 28 pixels.

The dataset has 10 classes, representing different clothing categories such as T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

Training set: 60,000 images.

Test set: 10,000 images.

Images are preprocessed (normalized and converted to tensors) before being passed into the CNN.


## Neural Network Model

<img width="1499" height="709" alt="image" src="https://github.com/user-attachments/assets/6daf0212-7d4c-4a0d-866c-1a4cb0026063" />


## Neural Network Model Explanation
## 1. Architecture
The model is a deep convolutional neural network consisting of convolutional, pooling, and fully connected layers.

Convolutional Layer 1

Input channels: 1 (grayscale images).

Output channels: 16 feature maps.

Kernel size: 3×3 with padding=1.

Activation: ReLU.

Followed by AvgPool2d(2×2) to reduce spatial dimensions.

Convolutional Layer 2

Input channels: 16 → Output channels: 32.

Kernel size: 3×3, padding=1.

Activation: ReLU.

Followed by AvgPool2d(2×2).

Convolutional Layer 3

Input channels: 32 → Output channels: 64.

Kernel size: 3×3, padding=1.

Activation: ReLU.

Followed by AvgPool2d(2×2).

Convolutional Layer 4

Input channels: 64 → Output channels: 128.

Kernel size: 3×3, padding=1.

Activation: ReLU.

(No pooling here, preserving feature richness).

Flatten Layer

Flattens the feature maps into a 1D vector: size 128 × 3 × 3 = 1152.

Fully Connected Layer 1: 1152 → 256 (ReLU).

Fully Connected Layer 2: 256 → 128 (ReLU).

Fully Connected Layer 3: 128 → 64 (ReLU).

Output Layer: 64 → 10 (one neuron per class).

Flow of data:

Input (1×28×28) → Conv1 → Pool → Conv2 → Pool → Conv3 → Pool → Conv4 → Flatten → FC1 → FC2 → FC3 → FC4 → Output (10 classes)

## 2. Activation Functions
ReLU is used after each convolutional and fully connected layer to introduce non-linearity.

The final output layer produces logits (raw scores). During training, CrossEntropyLoss internally applies Softmax + log, so no explicit softmax is added in the forward pass.

## 3. Loss Function
CrossEntropyLoss is used.

It compares the predicted logits with the true labels of 10 clothing categories.

Standard choice for multi-class classification tasks.

## 4. Optimizer
Adam optimizer is chosen.

Efficient and adaptive learning rate.

Provides faster convergence for deep CNNs.

## DESIGN STEPS

### STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### STEP 2:
Load the FashionMNIST dataset and apply transformations (normalization, tensor conversion).

### STEP 3:
Split the dataset into training and testing sets.

### STEP 4:
Define the CNN architecture with convolutional, pooling, and fully connected layers.

### STEP 5:
Specify the loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 6:
Train the model using forward pass, loss computation, backpropagation, and parameter updates.

### STEP 7:
Evaluate the model on the test dataset and calculate accuracy.

### STEP 8:
Test the trained model on new/unseen FashionMNIST images.

## PROGRAM

### Name: SANTHOS KUMAR R
### Register Number:212223240153
```python

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: Santhosh kumar R')
        print('Register Number: 212223240153')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="385" height="220" alt="image" src="https://github.com/user-attachments/assets/1b5b57aa-1003-4377-bc12-c5fe1329ad05" />

### Confusion Matrix

<img width="710" height="628" alt="image" src="https://github.com/user-attachments/assets/2c511f94-e7a8-4942-aab8-2535c9de8374" />

### Classification Report

<img width="579" height="325" alt="image" src="https://github.com/user-attachments/assets/5be5b210-8cd0-4bb1-b7e3-670a6a1082b9" />


### New Sample Data Prediction

<img width="572" height="456" alt="image" src="https://github.com/user-attachments/assets/fa95c293-d7f7-4986-bd1a-84f902b94a59" />

## RESULT
The Convolutional Neural Network was successfully implemented for FashionMNIST image classification. The model achieved good accuracy on the test dataset and produced reliable predictions for new images, proving its effectiveness in extracting spatial features from images.
