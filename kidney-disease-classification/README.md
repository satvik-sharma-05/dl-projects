
---

# 🩺 KidneyScan AI

### End-to-End Kidney Disease Classification using Deep Learning & MLOps

---

## 📌 Project Overview

**KidneyScan AI** is a complete **end-to-end Deep Learning + MLOps project** built to classify **kidney CT scan images** into:

* ✅ **Normal**
* 🚨 **Tumor**

This project was designed to **simulate a real-world industry ML workflow**, covering everything from:

* Data ingestion
* Model training
* Model evaluation
* Experiment tracking
* Pipeline automation
* Version control
* Web-based deployment

> ⚠️ **Disclaimer**
> This project is strictly for **educational and research purposes only**.
> It is **NOT a medical diagnostic tool** and must not be used for clinical decisions.

---

## 🎯 Why I Built This Project

I built this project to:

* Understand how **real ML projects are structured**
* Learn **MLOps practices** (DVC, MLflow)
* Build an **end-to-end deployable AI system**
* Practice **responsible AI decision logic**
* Create a **resume-ready project** for AI / ML roles

Instead of stopping at model training, I intentionally took this project **from notebook → pipeline → deployment**, exactly how it happens in real companies.

---

## 🧠 Problem Statement

Kidney tumors are often detected using **CT scan imaging**.
Manual inspection is time-consuming and requires expert radiologists.

The goal of this project is to:

> Automatically classify kidney CT scans as **Normal** or **Tumor** using a deep learning model, while also providing **confidence-based predictions** to avoid overconfident medical claims.

---

## 📂 Dataset Description

* Medical CT scan images of kidneys
* Two classes:

  * `Normal`
  * `Tumor`
* Images organized in folder structure compatible with CNN training
* Downloaded automatically using **Google Drive + gdown**

📁 Final structure after ingestion:

```
kidney-ct-scan-image/
├── normal/
└── tumor/
```

---

## 🏗️ How I Built This Project (Step-by-Step)

This section explains **exactly how I built the project**, in the same order I worked on it.

---

## 🔹 Step 1: Project Structure & Configuration

I first created a **clean, scalable folder structure**, separating:

* configuration
* components
* pipelines
* utilities

```
src/kidney_disease_classification/
├── components/      # Core ML logic
├── pipeline/        # Stage-wise pipelines
├── utils/           # Reusable helpers
├── config/          # Configuration manager
├── entity/          # Config dataclasses
```

Why this matters:

* Makes the project **maintainable**
* Allows **easy extension**
* Follows **industry ML standards**

---

## 🔹 Step 2: Configuration Management (YAML + Dataclasses)

I used:

* `config.yaml` → paths & artifacts
* `params.yaml` → model hyperparameters

Example:

```yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 8
EPOCHS: 20
```

I then used **Python dataclasses** to strongly type configurations, which:

* Prevents runtime bugs
* Makes configs explicit
* Improves readability

---

## 🔹 Step 3: Data Ingestion Pipeline (Stage 01)

📄 `stage_01_data_ingestion.py`

What this stage does:

* Downloads dataset from Google Drive
* Extracts ZIP file
* Stores data in `artifacts/data_ingestion/`

Key tools used:

* `gdown`
* `zipfile`
* automated directory creation

Why this matters:

* Fully **reproducible data pipeline**
* No manual downloads
* DVC can track data changes

---

## 🔹 Step 4: Base Model Preparation (Stage 02)

📄 `stage_02_prepare_base_model.py`

What I did:

* Used **Transfer Learning**
* Loaded a pretrained CNN base
* Configured:

  * input shape
  * number of classes
  * learning rate
* Saved the base model as an artifact

Why:

* Transfer learning gives **better accuracy**
* Faster training
* Less data required

---

## 🔹 Step 5: Model Training (Stage 03)

📄 `stage_03_model_training.py`

This stage:

* Loads the prepared base model
* Creates image generators
* Applies rescaling
* Trains the model
* Saves final trained model as:

```
artifacts/training/model.h5
```

Training results:

* Accuracy reached **~100% on validation**
* Loss reduced significantly

Why I saved the model as an artifact:

* Enables versioning
* Enables deployment
* Enables rollback if needed

---

## 🔹 Step 6: Model Evaluation + MLflow (Stage 04)

📄 `stage_04_model_evaluation.py`

This stage:

* Loads trained model
* Runs evaluation on validation data
* Saves metrics to `scores.json`
* Logs:

  * parameters
  * metrics
  * model
* Uses **MLflow** for experiment tracking

Example metrics:

```json
{
  "loss": 0.041,
  "accuracy": 1.0
}
```

Why MLflow:

* Tracks experiments
* Enables comparison
* Shows professional MLOps skills

---

## 🔹 Step 7: Pipeline Automation with DVC

I connected all stages using `dvc.yaml`:

Stages:

1. data_ingestion
2. prepare_base_model
3. training
4. evaluation

Command used:

```bash
dvc repro
```

Why DVC:

* Reproducibility
* Dependency tracking
* Pipeline automation
* Industry-standard MLOps tool

---

## 🔹 Step 8: Prediction Pipeline

📄 `prediction.py`

What it does:

* Loads trained model
* Preprocesses uploaded image
* Outputs:

  * prediction label
  * confidence score
  * class probabilities

Important logic I added:

### ✅ Confidence Threshold Logic

| Confidence | Output         |
| ---------- | -------------- |
| ≥ 70%      | Normal / Tumor |
| < 70%      | Uncertain      |

Why:

* Medical AI must be cautious
* Prevents false confidence
* Shows **responsible AI design**

---

## 🔹 Step 9: Streamlit Web Application

📄 `app.py`

Features:

* Image upload
* Live prediction
* Confidence display
* Explanation of results
* Clear UI states:

  * Normal
  * Tumor
  * Uncertain
* Medical disclaimer

UI highlights:

* Confidence bars
* Threshold explanation
* User-friendly messaging

---

## 🔹 Step 10: Deployment on Streamlit Cloud

Steps:

1. Pushed project to GitHub
2. Created `requirements.txt`
3. Deployed using Streamlit Cloud
4. App runs fully online for free

Why Streamlit:

* Fast
* Free
* ML-friendly
* No DevOps complexity

---

## 📊 Model Performance Summary

| Metric   | Value         |
| -------- | ------------- |
| Accuracy | ~100%         |
| Loss     | ~0.04         |
| Classes  | Normal, Tumor |

⚠️ Note:

* Trained on limited dataset
* Real-world performance may vary

---

## ⚠️ Medical Disclaimer (Very Important)

This project:

* ❌ Is NOT a medical diagnostic system
* ❌ Is NOT clinically validated
* ❌ Should NOT be used for treatment decisions

Always consult certified medical professionals.

---

## 🧠 What This Project Demonstrates

* Deep Learning (CNNs)
* Transfer Learning
* Data Pipelines
* MLOps (DVC + MLflow)
* Experiment tracking
* Responsible AI
* End-to-end deployment
* Clean software engineering practices

---

## 👨‍💻 Author

**Satvik Sharma**

AI / ML Engineer

Deep Learning • MLOps • Deployment

