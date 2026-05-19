# 🫁 Pneumonia Detection from Chest X-Rays
### A Comparative Study of Machine Learning vs. Deep Learning


---

## 📋 Table of Contents
- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation & Usage](#installation--usage)
- [References](#references)

---

## Overview

This project presents a comparative study between traditional **Machine Learning (ML)** and modern **Deep Learning (DL)** approaches for binary classification of chest X-ray images into **Normal** and **Pneumonia** classes.

**Four models** were implemented and evaluated:
| Approach | Models |
|----------|--------|
| Machine Learning | HOG + SVM, HOG + KNN |
| Deep Learning | VGG16 (Transfer Learning), ResNet50 (Transfer Learning) |

---

## Results at a Glance

| Model | Type | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|--------|----------|
| HOG + SVM | ML | 76.60% | 85.00% | 69.00% | 70.00% |
| HOG + KNN | ML | 78.53% | 81.00% | 73.00% | 74.00% |
| ResNet50 | DL | 87.02% | 90.13% | 88.97% | 89.55% |
| **VGG16** | **DL** | **91.51%** | **90.41%** | **96.67%** | **93.43%** |

> 🏆 **VGG16** achieved the best performance with **91.51% accuracy** and **93.43% F1-score**.

---

## Dataset

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** from Kaggle.

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Training | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> ⚠️ Notable class imbalance: ~73% of training images are Pneumonia cases. This was addressed using balanced class weights (SVM) and data augmentation (DL models).

---

## Project Structure

```
📦 pneumonia-detection/
├── 📓 ml_hog_svm_knn.ipynb        # ML models: HOG + SVM & HOG + KNN
├── 📓 dl_resnet50.ipynb            # DL model: ResNet50 with transfer learning
├── 📓 dl_vgg16.ipynb     # DL model: VGG16 with transfer learning
└── 📄 README.md
```

---

## Methodology

### 🔷 Machine Learning Pipeline (`ml_hog_svm_knn.ipynb`)

1. **Preprocessing** — Grayscale conversion, resize to 128×128
2. **Feature Extraction** — HOG (Histogram of Oriented Gradients)
   - `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`
3. **Classifiers**
   - **SVM**: RBF kernel, balanced class weights
   - **KNN**: k=5, Euclidean distance

### 🔶 Deep Learning Pipeline (`dl_resnet50.ipynb`, `dl_vgg16.ipynb`)

1. **Preprocessing** — Resize to 224×224, normalize to [0,1]
2. **Data Augmentation** — Rotation (15°), width/height shift (0.1), zoom (0.15), horizontal flip, brightness [0.9–1.1]
3. **Architecture** — Pretrained base + custom head:
   ```
   GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
   ```
4. **Two-Phase Training Strategy**

   | Phase | Description | Epochs | LR |
   |-------|-------------|--------|----|
   | Phase 1 | Feature Extraction (base frozen) | 10 | 0.0001 |
   | Phase 2 | Fine-Tuning (top layers unfrozen) | 10 | 0.00001 |

   - **VGG16**: First 15 layers frozen during fine-tuning
   - **ResNet50**: First 140 layers frozen during fine-tuning
   - Both use: Binary cross-entropy, Adam optimizer, EarlyStopping, ReduceLROnPlateau

---

## Installation & Usage

### Prerequisites
```bash
pip install tensorflow scikit-learn scikit-image opencv-python kagglehub seaborn matplotlib numpy
```

### Running the Notebooks

**1. Clone the repository**
```bash
git clone https://github.com/Elsebae/Pneumonia-Detection-from-Chest-X-Rays
cd pneumonia-detection
```

**2. Set up Kaggle API** (for automatic dataset download)
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**3. Run the notebooks**
- Open `ml_hog_svm_knn.ipynb` for ML models (CPU sufficient)
- Open `dl_resnet50.ipynb` for ResNet50 (GPU recommended)
- Open `dl_vgg16.ipynb` for VGG16 (GPU recommended)

> 💡 DL models were trained for 20 epochs each and require a GPU for reasonable training time.

---

## Key Findings

- ✅ Deep learning models **significantly outperformed** ML models — VGG16 achieved 12.98% higher accuracy than the best ML model
- ✅ DL models showed **balanced performance** across both classes; ML models were biased toward predicting Pneumonia
- ✅ **VGG16 outperformed ResNet50**, showing that architectural complexity doesn't always win
- ✅ **Transfer learning with fine-tuning** proved highly effective for medical image classification
- ✅ Automatic feature learning (CNNs) is **superior to handcrafted features** (HOG) for this task

---

## References

1. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, 2017.
2. Liu et al., "From BoW to CNN: Two decades of texture representation," *IJCV*, 2019.
3. LeCun, Bengio, Hinton, "Deep learning," *Nature*, 2015.
4. Pan & Yang, "A survey on transfer learning," *IEEE TKDE*, 2010.
5. Kermany et al., "Identifying medical diagnoses by image-based deep learning," *Cell*, 2018.
6. Dalal & Triggs, "Histograms of oriented gradients for human detection," *CVPR*, 2005.
7. Simonyan & Zisserman, "Very deep convolutional networks," *ICLR*, 2015.
8. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
