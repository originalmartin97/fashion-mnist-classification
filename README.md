# Fashion-MNIST Classification

A deep learning project for classifying clothing items using the Fashion-MNIST dataset with PyTorch.

## 📋 Project Overview

This project implements a neural network classifier to recognize 10 different types of clothing items from grayscale images. The model achieves ~88% test accuracy using custom normalization and optimized training strategies (up to now).

## 🗂️ Dataset

**Fashion-MNIST** contains 70,000 grayscale images (28×28 pixels):
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Classes:** 10 clothing categories

| Class | Label |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🏗️ Project Structure

```
fashion-mnist-classification/
├── .gitignore                                  # Git ignore rules
├── archive/                                    # Previous experimental
├── docs/                                       # Documentation files
├── notebooks/
│   ├── EDA_MNISTFashion_classification.ipynb   # Exploratory data analysis
│   ├── MNISTFashion_classifiaction.ipynb       # Main classification 
│   └── data/                                   # Notebook-specific data
├── README.md
└── requirements.txt                            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/originalmartin97/fashion-mnist-classification.git
cd fashion-mnist-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks:
```bash
jupyter notebook
```

## 🧠 Model Architecture

**FashionNet** - Fully connected neural network:
- Input layer: 784 neurons (28×28 flattened)
- Hidden layer 1: 128 neurons + ReLU + Dropout(0.2)
- Hidden layer 2: 64 neurons + ReLU + Dropout(0.2)
- Output layer: 10 neurons (classification)

## 📊 Results

- **Test Accuracy:** ~88%
- **Training Strategy:** 10 epochs, Adam optimizer (lr=0.001)
- **Normalization:** Custom (mean=0.2913, std=0.3552)
- **Loss Function:** CrossEntropyLoss

## 📓 Notebooks

### 1. EDA_MNISTFashion_classification.ipynb
Comprehensive exploratory data analysis including:
- Dataset overview and statistics
- Sample visualizations
- Class distribution analysis
- Pixel value analysis

### 2. MNISTFashion_classifiaction.ipynb
Main classification notebook with:
- Data preprocessing and normalization
- Model architecture definition
- Training process and visualization
- Results analysis and evaluation

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **matplotlib** - Data visualization
- **NumPy** - Numerical computing

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**originalmartin97**

## 🙏 Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- PyTorch community and documentation
