# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name:
### Register Number:
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
