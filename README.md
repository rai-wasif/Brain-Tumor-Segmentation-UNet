# 🧠 Brain Tumor Segmentation (U-Net)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-green)

A semantic segmentation model designed to automatically identify Lower-Grade Gliomas (LGG) in brain MRI scans. This project implements a **U-Net architecture** from scratch in PyTorch, tackling extreme class imbalance to achieve high sensitivity (Recall) for medical diagnostics.

### 🔎 Model Predictions (Visualizing Confidence)
Below are examples of the model's performance on unseen test data during training. The red areas in the heatmap indicate high model confidence in the presence of a tumor.

<table>
  <tr>
    <td align="center"><img src="heatmap_results.png" alt="Result Example 1" width="100%"></td>
    <td align="center"><img src="heatmap-results1.png" alt="Result Example 2" width="100%"></td>
    <td align="center"><img src="heatmap-results2.png" alt="Result Example 3" width="100%"></td>
  </tr>
</table>
<p align="center"><i>(Within each image above: Left=Original MRI, Middle=Ground Truth Label, Right=Model Prediction Heatmap)</i></p>

---

## 📌 Project Overview

**The Challenge:**
Manual segmentation of brain tumors is time-consuming and prone to inter-observer variability. Additionally, medical datasets are heavily imbalanced—tumor pixels often represent less than 1% of the image, causing standard models to bias towards the background (99% accuracy but 0% tumor detection).

**The Solution:**
I engineered a deep learning pipeline that prioritizes **Recall** (finding the cancer) over pure accuracy. By using a custom **Dice Loss** function and heavy data augmentation, the model learns to delineate exact tumor boundaries despite the small size of the targets.

## 🌐 Deployment & MLOps (FastAPI + Streamlit)

To bridge the gap between research and production, I deployed the model using a microservices architecture.

### ⚡ Backend: Why FastAPI?
I chose **FastAPI** over Flask/Django for serving the model due to:
* **Asynchronous Performance:** FastAPI supports `async`/`await`, allowing the server to handle multiple inference requests simultaneously without blocking (crucial for heavy deep learning tasks).
* **Data Validation:** Automatic validation using Python type hints ensures that image payloads are correctly formatted before they reach the model.
* **Auto-Documentation:** Generates interactive Swagger UI (`/docs`) automatically, simplifying API testing.

### 🖥️ Frontend: Live Demo via Streamlit
A user-friendly web interface was built using **Streamlit** to allow radiologists/users to upload MRI scans and get real-time segmentation overlays.

**📸 Real-Time Testing Screenshots:**
<table>
  <tr>
    <td align="center"><b>Test Case 1</b><br><img src="testcase.png" width="100%"></td>
    <td align="center"><b>Test Case 2</b><br><img src="testcase1.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Test Case 3</b><br><img src="testcase2.png" width="100%"></td>
    <td align="center"><b>Test Case 4</b><br><img src="testcase3.png" width="100%"></td>
  </tr>
</table>

---

## 📂 Dataset

**LGG MRI Segmentation Dataset**
* **Source:** [Kaggle - Mateusz Buda](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
* **Content:** 110 patients, ~3,900 MRI slices (RGB) with corresponding manual segmentation masks.
* **Preprocessing:** Images resized to 256x256 and normalized.

## ⚙️ Methodology & Strategy

### 1. Model Architecture: U-Net
I implemented the U-Net architecture (Ronneberger et al.) from scratch.
* **Encoder:** Captures high-level context (what is the object?).
* **Decoder:** Recovers spatial resolution (where is the object?).
* **Skip Connections:** Concatenate encoder features with decoder features to preserve fine edge details lost during downsampling.

### 2. Handling Class Imbalance (The "Dice" Strategy)
Standard Cross-Entropy loss fails here because the background dominates the loss function.
* **Loss Function:** `BCEWithLogitsLoss + DiceLoss`
* **Why?** Dice Loss directly optimizes the **overlap (Intersection over Union)** between prediction and ground truth, ignoring the massive black background.

### 3. Data Augmentation (Albumentations)
To prevent overfitting on the small dataset (110 patients), I applied:
* Random Rotations (±35°)
* Horizontal & Vertical Flips
* This forces the model to learn tumor *features* rather than memorizing *locations*.

## 📊 Results (After 28 Epochs)

The model was evaluated on a held-out test set.

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **IoU (Intersection over Union)** | **0.7257** | High overlap with doctor's manual segmentation. |
| **Recall (Sensitivity)** | **0.8398** | **Crucial:** The model detects **84%** of all tumor tissue. |
| **Precision** | **0.8733** | Very low rate of false alarms. |
| **F1-Score** | **0.8562** | Excellent harmonic mean of precision and recall. |

### Classification Report (Pixel-wise)
```text
              precision    recall  f1-score   support

  Background     0.9984    0.9988    0.9986  25504188
       Tumor     0.8733    0.8398    0.8562    251460

    accuracy                         0.9972  25755648
   macro avg     0.9359    0.9193    0.9274  25755648
weighted avg     0.9972    0.9972    0.9972  25755648


###   🛠️ Tech Stack
'''text
 Deep Learning: PyTorch, Albumentations

Deployment: FastAPI (Backend), Streamlit (Frontend), Uvicorn

Computer Vision: OpenCV, PIL

Environment: Kaggle Kernels (Training), Local GPU (Inference)

**🚀 How to Run
1. Clone the repository
Bash

git clone https://github.com/rai-wasif/Brain-Tumor-Segmentation-UNet.git
cd Brain-Tumor-Segmentation-UNet
2. Run the Web App (Deployment)
Navigate to the deployment folder and install dependencies:

Bash

cd deployment
pip install -r requirements.txt
Start the Backend (FastAPI):

Bash

uvicorn main:app --reload
Start the Frontend (Streamlit) in a new terminal:

Bash

streamlit run frontend.py
3. Training the Model (Optional)
If you wish to retrain the model from scratch:

Download the dataset from Kaggle.

Update the path in brain-mri-unet.ipynb.

Run the Jupyter Notebook.

👤 Author
M WASIF YASEEN

GitHub: rai-wasif

Email: raimuhammadwasif@gmail.com**
