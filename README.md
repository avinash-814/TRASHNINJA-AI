
#  SmartBin – AI Waste Classification Model

This folder contains the trained **Convolutional Neural Network (CNN)** model used in the **SmartBin** prototype to classify plastic waste as either **Recyclable** or **Non-Recyclable**.


# Folder Structure

ai_model/
├── cnn_train.py        # Script to train the CNN model (requires dataset)
├── cnn_predict.py      # Script to perform prediction using trained model
├── smartbin_cnn_model.keras   # Pre-trained CNN model



##  Model Overview

- **Model Type:** Convolutional Neural Network (CNN)
- **Classification Categories:** 
  - Recyclable 
  - Non-Recyclable 



##  How to Use

 1. Predict using the trained model

   bash
python cnn_predict.py


- This script will load the `smartbin_cnn_model.keras` file and perform predictions on sample images.
- You can modify the script to capture frames from your webcam and show predictions live.


 2. Train the model (Optional)

  > ⚠️ Dataset and smartbin_cnn_model.keras are not included (due to GitHub upload size limits)


If you want to re-train the model:

- Make sure your folder structure is:

dataset/
├── recyclable/
│   └── image1.jpg ...
├── nonrecyclable/
│   └── image2.jpg ...


- Then run:
```bash
python cnn_train.py
```

---

## 📦 Dataset

The dataset used to train this model is not uploaded due to GitHub's size limitations (25MB max).  
It contains:
- `recyclable/` – images of recyclable waste  
- `nonrecyclable/` – images of non-recyclable waste  

📩 **Dataset can be shared upon request** or generated manually by collecting labeled waste images.

---

## 🖼️ Sample Prediction Output

```text
Prediction: Recyclable (97.34%)
```

- The model shows the classification result along with confidence percentage.

---

## 👨‍💻 Developed By techsquad

**likhitha sunkari** 
**Avinash Narsipuram**  
**simhachalam murabanda** 
**komali pilla** 
TRASHNINJA (a smartbin) Project – SPINFESTHackathon 2025  
[GitHub Profile](https://github.com/avinash-814)


## 💡 Project Use Case

SmartBin aims to help users **identify the type of plastic** in real-time using AI-powered classification.  
It enhances the waste sorting process and encourages environmental responsibility.

---
