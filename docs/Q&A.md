# Dataset Preparation

## Data Loaders
### **Why normalize?**
Normalization centers the data around zero and scales it to unit variance, which:
- Speeds up training convergence
- Prevents gradient vanishing/exploding
- Improves model stability


### **shuffle**

**What it does:** Randomizes the order of data each epoch.
```py
# Training: shuffle=True
Epoch 1: [img_42, img_7, img_103, img_56, ...]  # Random order
Epoch 2: [img_91, img_3, img_77, img_12, ...]   # Different random order

# Testing: shuffle=False  
Always:  [img_0, img_1, img_2, img_3, ...]      # Same order every time
```

### **num_workers**
**What it does:** Number of background processes loading data in parallel.
```py
num_workers=0:  Load batch → Train → Load batch → Train → ...
                (sequential, slow)

num_workers=2:  [Worker 1 loads next batch]
                [Worker 2 loads batch after that]
                     ↓
                Main process trains on current batch
                (parallel, faster!)
```

### **What is PIL?**
PIL = Python Imaging Library (now called Pillow)

It's a standard Python library for opening, manipulating, and saving images.

```py
from PIL import Image

img = Image.open("sneaker.jpg")  # PIL Image object
```
When PyTorch's torchvision loads Fashion-MNIST, each image starts as a PIL Image—basically a regular image file in memory.

### **Why Convert to Tensors?**

**1. Neural networks need numbers, not images**
```py
PIL Image:  "A picture of a sneaker" (image file format)
     ↓
Tensor:     [[0.12, 0.45, 0.78, ...], [0.23, 0.56, ...], ...]  (numbers!)
```

**2. GPU acceleration**
```py
tensor.to('cuda')  # Move to GPU for fast parallel computation
# PIL images can't do this!
```

**3. Automatic gradients (backpropagation)**
```py
loss.backward()  # PyTorch tracks gradients through tensors
# Only works with tensors, not PIL images
```

**4. Batch processing**
```py
# Stack 64 tensors into one batch
batch = torch.stack([tensor1, tensor2, ..., tensor64])
# Shape: (64, 1, 28, 28) - process all at once!
```

