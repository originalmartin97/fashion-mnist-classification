# Project Structure Explained

Understanding how the Fashion-MNIST classification project builds up, block by block.

---

## **Block 1: Import Libraries**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**What it does:**
- Imports all the tools you need for deep learning
- `torch` = PyTorch core library (tensors, operations)
- `nn` = Neural network building blocks (layers, loss functions)
- `optim` = Optimizers (algorithms that update weights)
- `torchvision` = Computer vision datasets and tools
- `transforms` = Data preprocessing tools
- `matplotlib` = Plotting graphs
- `device` = Checks if you have a GPU (CUDA), otherwise uses CPU

**What you can tweak:**
- Nothing really, these are standard imports

---

## **Block 2: Load Data**

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2913,), (0.3552,))
])
```

**What it does:**
- **ToTensor()** = Converts images from PIL format to PyTorch tensors (0-1 range)
- **Normalize()** = Scales pixel values using mean and std deviation
  - Formula: `pixel_normalized = (pixel - 0.2913) / 0.3552`
  - Makes training more stable and faster

**What you can tweak:**
- **Normalization values** (0.2913, 0.3552) - these are Fashion-MNIST specific
- **Add more transforms**: `transforms.RandomRotation(10)`, `transforms.RandomHorizontalFlip()`

---

```python
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

**What it does:**
- Downloads Fashion-MNIST dataset to `./data` folder
- Applies the transformations (ToTensor + Normalize)
- `train=True` = Get training set (60,000 images)

**What you can tweak:**
- `root='./data'` = Change download location

---

```python
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)
```

**What it does:**
- **DataLoader** = Feeds data to the model in batches
- **batch_size=64** = Process 64 images at once (not all 60,000)
  - Smaller batch = slower but more accurate updates
  - Larger batch = faster but less precise updates
- **shuffle=True** = Randomize order each epoch (prevents memorization)

**What you can tweak:**
- **batch_size**: Try 32, 64, 128, 256
- **shuffle**: Keep True for training, False for testing

---

## **Block 3: Define Model**

```python
class FashionNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(FashionNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)   # Hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer
        self.relu = nn.ReLU()           # Activation
        self.dropout = nn.Dropout(dropout)  # Regularization
```

**What each layer does:**

| Layer | Input → Output | Purpose |
|-------|----------------|---------|
| `fc1` | 784 → 128 | Takes flattened image (28×28=784 pixels), outputs 128 features |
| `fc2` | 128 → 64 | Compresses to 64 features |
| `fc3` | 64 → 10 | Final layer outputs 10 scores (one per clothing class) |
| `relu` | Activation | Adds non-linearity: `f(x) = max(0, x)` |
| `dropout` | Regularization | Randomly zeros 20% of neurons during training |

**What you can tweak:**
- **Layer sizes**: `nn.Linear(784, 256)` = more neurons, more capacity
- **Number of layers**: Add `self.fc4 = nn.Linear(64, 32)` for deeper network
- **Dropout rate**: `dropout=0.3` or `0.4` = more regularization
- **Activation function**: `nn.LeakyReLU()`, `nn.Tanh()` instead of ReLU

---

```python
def forward(self, x):
    x = x.view(-1, 784)  # Flatten image
    x = self.dropout(self.relu(self.fc1(x)))
    x = self.dropout(self.relu(self.fc2(x)))
    x = self.fc3(x)
    return x
```

**What it does:**
- **Data flow through the network:**
  1. Flatten: [64, 1, 28, 28] → [64, 784]
  2. fc1 → ReLU → Dropout: [64, 784] → [64, 128]
  3. fc2 → ReLU → Dropout: [64, 128] → [64, 64]
  4. fc3: [64, 64] → [64, 10]
  5. Return 10 scores (logits)

**Why no activation on fc3?**
- CrossEntropyLoss applies softmax internally
- We return raw scores (logits)

---

## **Block 4: Training Setup**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
```

**What each does:**

**CrossEntropyLoss:**
- Measures how wrong predictions are
- Combines softmax + negative log likelihood
- Lower loss = better predictions

**Adam Optimizer:**
- Updates model weights to minimize loss
- `lr=0.001` = learning rate (step size)
  - Too high = unstable training, overshoots
  - Too low = slow training, gets stuck

**num_epochs:**
- How many times to go through entire training set
- 1 epoch = see all 60,000 images once

**What you can tweak:**
- **Loss function**: `nn.NLLLoss()`, `nn.BCELoss()` (for different tasks)
- **Optimizer**: `optim.SGD()`, `optim.RMSprop()`
- **Learning rate**: Try 0.0001, 0.001, 0.01
- **Epochs**: 5, 10, 15, 20 (watch for overfitting)

---

## **Block 5: Training Functions**

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Enable dropout
    running_loss = 0.0
    correct = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear old gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights
```

**Step-by-step what happens:**

1. **model.train()** = Turn on dropout for training
2. **Loop through batches** (64 images at a time)
3. **optimizer.zero_grad()** = Clear previous gradients (PyTorch accumulates them)
4. **Forward pass** = Run images through network, get predictions
5. **Calculate loss** = Compare predictions to true labels
6. **loss.backward()** = Calculate gradients (how to adjust weights)
7. **optimizer.step()** = Update weights using gradients

**Result:** Returns average loss and accuracy for the epoch

---

```python
def evaluate(model, loader, criterion, device):
    model.eval()  # Disable dropout
    
    with torch.no_grad():  # Don't calculate gradients
        # Same as train_epoch but no backward() or step()
```

**What's different:**
- **model.eval()** = Turns off dropout (use all neurons)
- **torch.no_grad()** = Saves memory, faster (no gradient calculation)
- **No optimization** = Just measure performance, don't update weights

---

## **Block 6: Training Loop**

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    test_loss, test_acc = evaluate(...)
    
    # Store metrics
    train_losses.append(train_loss)
    test_accs.append(test_acc)
```

**What happens each epoch:**
1. Train on all 60,000 training images (in batches)
2. Test on all 10,000 test images
3. Record loss and accuracy
4. Print progress
5. Repeat

**Purpose:** See how model improves over time

---

## **Block 7: Visualize Results**

```python
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
```

**What you see:**
- **Loss graph** = Should decrease over time (model learning)
- **Accuracy graph** = Should increase over time (better predictions)
- **Gap between train/test** = Shows overfitting
  - Small gap = good generalization ✓
  - Large gap = memorizing training data ✗

---

## **Summary: What Can You Tweak?**

| Component | Parameter | Effect |
|-----------|-----------|--------|
| **Data** | `batch_size` | Speed vs accuracy |
| **Data** | Normalization values | Training stability |
| **Model** | Layer sizes (128, 64) | Model capacity |
| **Model** | Number of layers | Model depth |
| **Model** | `dropout` rate | Regularization strength |
| **Training** | `lr` (learning rate) | Training speed |
| **Training** | `num_epochs` | Training duration |
| **Training** | Optimizer | Update strategy |

---

## **The Complete Workflow**

```
1. IMPORT TOOLS
   ↓ (Get PyTorch, data tools, plotting)
   
2. GET THE DATA
   ↓ (Download Fashion-MNIST, apply normalization)
   
3. CHECK THE DATA (EDA)
   ↓ (Visualize samples, understand what you're working with)
   
4. DEFINE THE MODEL
   ↓ (Create neural network architecture: layers, activations)
   
5. SETUP TRAINING
   ↓ (Choose loss function, optimizer, learning rate)
   
6. TRAIN THE MODEL
   ↓ (Loop: forward pass → calculate loss → backward pass → update weights)
   
7. EVALUATE
   ↓ (Test on unseen data, measure accuracy)
   
8. VISUALIZE RESULTS
   ↓ (Plot loss/accuracy curves, analyze performance)
```

**Core Cycle:** DATA → MODEL → TRAIN → EVALUATE → (repeat if needed)
