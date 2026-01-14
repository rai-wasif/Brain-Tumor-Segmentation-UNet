````markdown
# 🧠 Brain Tumor Segmentation (U-Net)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-green)

A semantic segmentation model designed to automatically identify **Lower-Grade Gliomas (LGG)** in brain MRI scans.  
The project implements a **U-Net architecture from scratch in PyTorch**, addressing extreme class imbalance to achieve **high recall**, which is critical for medical diagnostics.

---

## 🔎 Model Predictions (Visualizing Confidence)

Below are examples of model predictions on unseen test data.  
Red regions in the heatmaps indicate high confidence tumor areas.

<table>
  <tr>
    <td align="center"><img src="heatmap_results.png" width="100%"></td>
    <td align="center"><img src="heatmap-results1.png" width="100%"></td>
    <td align="center"><img src="heatmap-results2.png" width="100%"></td>
  </tr>
</table>

<p align="center"><i>Left: Original MRI | Middle: Ground Truth | Right: Prediction Heatmap</i></p>

---

## 📌 Project Overview

### The Challenge
Manual brain tumor segmentation is time-consuming and prone to observer variability.  
Medical imaging datasets are highly imbalanced — tumor pixels often represent **less than 1%** of the image, causing standard models to achieve high accuracy while missing tumors entirely.

### The Solution
This project prioritizes **Recall (Sensitivity)** over raw accuracy by:
- Using a **Dice-based loss function**
- Applying strong data augmentation
- Designing an end-to-end deployment-ready pipeline

---

## 🌐 Deployment & MLOps

### ⚡ Backend — FastAPI
FastAPI was selected due to:
- Asynchronous inference support
- Automatic request validation
- Interactive API documentation (`/docs`)

### 🖥️ Frontend — Streamlit
A Streamlit interface allows users to upload MRI scans and visualize tumor segmentation results in real time.

---

## 📂 Dataset

**LGG MRI Segmentation Dataset**
- **Source:** Kaggle (Mateusz Buda)
- **Patients:** 110
- **MRI Slices:** ~3,900
- **Preprocessing:** Resized to 256×256 and normalized

---

## ⚙️ Methodology

### 1. Model Architecture — U-Net
- Encoder captures contextual features
- Decoder restores spatial resolution
- Skip connections preserve tumor boundaries

### 2. Handling Class Imbalance
- **Loss Function:** `BCEWithLogitsLoss + DiceLoss`
- Dice Loss directly optimizes overlap and reduces background dominance

### 3. Data Augmentation
- Random rotations (±35°)
- Horizontal and vertical flips  
These techniques improve generalization and prevent overfitting.

---

## 📊 Results (After 28 Epochs)

| Metric | Score |
|------|------|
| **IoU** | **0.7257** |
| **Recall (Sensitivity)** | **0.8398** |
| **Precision** | **0.8733** |
| **F1-Score** | **0.8562** |

### Pixel-wise Classification Report

```text
              precision    recall  f1-score   support

  Background     0.9984    0.9988    0.9986  25504188
       Tumor     0.8733    0.8398    0.8562    251460

    accuracy                         0.9972  25755648
   macro avg     0.9359    0.9193    0.9274  25755648
weighted avg     0.9972    0.9972    0.9972  25755648
````

---

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, Albumentations
* **Deployment:** FastAPI, Streamlit, Uvicorn
* **Computer Vision:** OpenCV, PIL
* **Environment:** Kaggle (Training), Local GPU (Inference)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/rai-wasif/Brain-Tumor-Segmentation-UNet.git
cd Brain-Tumor-Segmentation-UNet
```

### 2. Run the Web Application

```bash
cd deployment
pip install -r requirements.txt
```

**Start Backend**

```bash
uvicorn main:app --reload
```

**Start Frontend (new terminal)**

```bash
streamlit run frontend.py
```

### 3. Train the Model (Optional)

1. Download the dataset from Kaggle
2. Update dataset paths in `brain-mri-unet.ipynb`
3. Run the notebook

---

## 👤 Author

**M Wasif Yaseen**
GitHub: [https://github.com/rai-wasif](https://github.com/rai-wasif)
Email: [raimuhammadwasif@gmail.com](mailto:raimuhammadwasif@gmail.com)

```

---

```
