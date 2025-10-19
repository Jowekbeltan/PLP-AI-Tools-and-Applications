````markdown
# AI for Sustainable Development – Assignments

## 📘 Overview
This repository contains two major AI-related assignments:

1. **SDG 3 Project (Health & Well-being)** – Disease Prediction using Supervised Learning (Scikit-learn).
2. **AI Practical Exercises** – covering Classical ML, Deep Learning, and NLP using Python libraries.

---

## 🧩 Assignment 1: SDG 3 – Good Health and Well-being

### 🎯 Objective
Predict whether a person is at risk of a disease (e.g., diabetes or heart disease) using machine learning.

### 🧠 Approach
- **SDG Chosen:** SDG 3 – Good Health and Well-being  
- **Problem:** Early disease prediction using health data  
- **Machine Learning Type:** Supervised Learning  
- **Algorithm:** Logistic Regression / Decision Tree  
- **Dataset Source:** [Kaggle Health Datasets](https://www.kaggle.com/datasets)  
- **Tools:** Python, Jupyter Notebook, Scikit-learn  

### ⚙️ Workflow
1. Load and clean dataset (handle missing data)
2. Split into training and testing sets
3. Train a classification model
4. Evaluate using accuracy, precision, recall
5. Visualize predictions using matplotlib
6. Reflect on ethical and bias considerations

---

## 🤖 Assignment 2: AI Practical Tasks

### **Part 1 – Theoretical Understanding**
- Compared **TensorFlow vs PyTorch**
- Described use cases for **Jupyter Notebooks**
- Explained how **spaCy** enhances NLP tasks
- Compared **Scikit-learn vs TensorFlow**

---

### **Part 2 – Practical Implementation**

#### 🧮 Task 1: Classical ML (Scikit-learn)
- **Dataset:** Iris Species  
- **Goal:** Predict flower species using Decision Tree  
- **Evaluation:** Accuracy, Precision, Recall  

#### 🧠 Task 2: Deep Learning (TensorFlow)
- **Dataset:** MNIST Handwritten Digits  
- **Goal:** Train CNN model to classify digits  
- **Requirement:** Achieve >95% accuracy  
- **Output:** Visualize predictions on sample images  

#### 💬 Task 3: NLP with spaCy + TextBlob
- **Dataset:** Amazon Product Reviews  
- **Goal:** 
  - Extract Named Entities (NER)
  - Perform Sentiment Analysis (Positive, Negative, Neutral)
- **Libraries:** spaCy, TextBlob

---

### **Part 3 – Ethics & Optimization**
- Discussed bias detection and mitigation using:
  - `TensorFlow Fairness Indicators`
  - spaCy rule-based systems
- Debugged TensorFlow scripts with dimension mismatch and loss function errors.

---

## 🚀 Bonus Task
Deployed MNIST CNN model using **Streamlit** for real-time predictions (optional).

---

## 🛠️ Installation Guide

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (Mac/Linux)
````

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras spacy textblob
```

### 3. Download spaCy English Model

```bash
python -m spacy download en_core_web_sm
```

---

## ⚠️ Troubleshooting Notes

### 🧩 TensorFlow Errors

**Problem:** `No module named 'tensorflow'`
**Fix:**

1. Ensure long path support on Windows
2. Reinstall TensorFlow:

   ```bash
   pip install --upgrade pip
   pip install tensorflow
   ```

---

### 💬 spaCy Errors

**Problem:** `OSError: [E050] Can't find model 'en_core_web_sm'`
**Fix:**

```bash
python -m spacy download en_core_web_sm
```

**Optional Auto-Download (inside notebook):**

```python
import spacy, subprocess
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
```

---

## 📄 Deliverables

* **Code:** Jupyter Notebook (`.ipynb`)
* **Report:** 1-page summary (SDG, ML method, results, ethics)
* **Presentation:** 5-min video/demo

---

## 👩‍💻 Author

**Name:** Johnson Ontweka
**Course:** AI for Software Engineering
**Institution:** PLP Academy / JKUAT
**Year:** 2025

```

