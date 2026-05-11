# 🫀 Heart Disease Predictor

A machine learning web application that predicts the likelihood of heart disease based on clinical patient data. Built with **Streamlit** and powered by four trained ML models.

🔗 **Live Demo**: [Deploy link here]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Known Issues](#known-issues)

---

## Overview

This app allows healthcare professionals or researchers to input patient clinical data and receive heart disease predictions from four different machine learning models simultaneously. It supports both **single patient prediction** and **bulk CSV prediction**.

---

## ✨ Features

- **Single Prediction** — Enter 11 clinical features and get predictions from all 4 models at once
- **Bulk Prediction** — Upload a CSV file to predict heart disease for multiple patients in one go
- **Model Comparison** — Visual bar chart comparing the accuracy of all models
- **Downloadable Results** — Export bulk predictions as a CSV file

---

## 📊 Dataset

The app was trained on the **Heart Failure Prediction Dataset** (`heart.csv`), which contains 11 clinical features:

| Feature | Description |
|---------|-------------|
| `Age` | Age of the patient (years) |
| `Sex` | Sex of the patient (`Male` / `Female`) |
| `ChestPainType` | Chest pain type (`Typical Angina`, `Atypical Angina`, `Non-Anginal Pain`, `Asymptomatic`) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar (`1` if > 120 mg/dl, else `0`) |
| `RestingECG` | Resting ECG results (`Normal`, `ST-T Wave Abnormality`, `Left Ventricular Hypertrophy`) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (`1` = Yes, `0` = No) |
| `Oldpeak` | ST depression induced by exercise (float) |
| `ST_Slope` | Slope of peak exercise ST segment (`Upsloping`, `Flat`, `Downsloping`) |

**Target variable:** `HeartDisease` — `1` = Heart disease present, `0` = No heart disease

---

## 🤖 Models

Four machine learning models were trained and saved as `.pkl` files:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **85.86%** |
| Random Forest | 84.23% |
| Support Vector Machine | 84.22% |
| Decision Tree | 80.97% |

> Logistic Regression achieved the highest accuracy and is used for bulk predictions.

---

## 📁 Project Structure

```
heart-disease-predictor/
├── app.py                  # Main Streamlit application
├── heart.csv               # Training dataset
├── DecisionTree.pkl        # Trained Decision Tree model
├── RandomForest.pkl        # Trained Random Forest model
├── SVM.pkl                 # Trained SVM model
├── LogisticR.pkl           # Trained Logistic Regression model
├── heart_disease.png       # Image shown when disease is detected
├── healthy_heart.png       # Image shown when no disease is detected
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
   cd heart-disease-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 🖥️ Usage

### Single Prediction (Tab 1)
1. Enter the patient's clinical details using the input fields
2. Click the **Predict** button
3. View predictions from all 4 models with a final result indicator

### Bulk Prediction (Tab 2)
1. Prepare a CSV file with the 11 required columns (see dataset section for format)
2. Upload the CSV using the file uploader
3. View the predictions table and download the results

### Model Information (Tab 3)
- Read about each model used
- View the accuracy comparison bar chart

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud**.

To deploy your own instance:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `app.py`
5. Click **Deploy**

---

## ⚠️ Known Issues

- `heart_disease.png` and `healthy_heart.png` must be present in the repo root or the result images will not display
- Bulk prediction uses only the Logistic Regression model
- No input validation for medically implausible values (e.g. cholesterol = 0)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — Web app framework
- [scikit-learn](https://scikit-learn.org/) — Machine learning models
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computation
- [Plotly](https://plotly.com/) — Interactive charts

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
