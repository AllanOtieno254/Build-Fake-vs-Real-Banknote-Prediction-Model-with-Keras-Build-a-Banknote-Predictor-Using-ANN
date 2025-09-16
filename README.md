# 🏦 Fake vs Real Banknote Prediction using Deep Learning (ANN + Keras)
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6c8c7efd-c5b9-4f64-b525-d6f4602a5ddd" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d06690ba-4819-4df2-a843-77accc76761b" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c202789d-4a25-4954-a022-0a2b00a454fd" />


![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red.svg)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey.svg)
![Hackathon](https://img.shields.io/badge/Hackathon-Ready-success.svg)

---

## 📌 Project Overview  

This project implements a **Fake vs Real Banknote Classification System** using a **Deep Learning Artificial Neural Network (ANN)**.  
The system is capable of predicting whether a banknote is genuine or counterfeit based on key input features.  

It combines the power of **Machine Learning, Deep Learning, and Web Deployment (Flask)** to deliver an end-to-end solution:  

- 📊 **Data Preprocessing & Cleaning**  
- 🧠 **Artificial Neural Network built with Keras & TensorFlow**  
- 📈 **Model Training & Evaluation**  
- 🌐 **Web Deployment using Flask**  
- ⚡ **Real-time Prediction of Banknotes**

This project is designed not only as a machine learning exercise but also as a **hackathon-ready, production-style application** that demonstrates **data science, AI, and software engineering skills**.  

---

## 🎯 Motivation  

Fake currency is a serious issue affecting economies worldwide. Detecting counterfeit banknotes with precision is crucial for banks, governments, and businesses.  

This project leverages **AI-powered ANN models** to provide:  
✅ Automated Detection  
✅ Higher Accuracy than Traditional Methods  
✅ A Scalable Solution deployable via Web & APIs  

By building this project, I showcase my ability to take an idea from **data to model to deployment**, which is a critical skill for modern AI engineers.  

---

## 🧠 How It Works  

The workflow follows these steps:  

1. **Dataset Import** – The dataset contains features extracted from banknotes.  
   Features include statistical measures such as variance, skewness, curtosis, and entropy.  

2. **Data Preprocessing** – Missing values handled, data normalized, and split into training/test sets.  

3. **Model Building (Keras ANN)** –  
   - Input Layer (features)  
   - Hidden Layers (ReLU activation, Dropout)  
   - Output Layer (Sigmoid for binary classification)  

4. **Training** – Model trained on labeled dataset with backpropagation & optimization.  

5. **Evaluation** – Performance measured using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  

6. **Deployment (Flask)** – Exposes a simple **web UI + API endpoint** where users can input features and get predictions in real-time.  

---

## 🛠️ Tech Stack & Dependencies  

The project is powered by:  

- **Python 3.13+**
- **TensorFlow 2.x**
- **Keras (built-in TensorFlow)**
- **NumPy** – numerical computations  
- **Pandas** – data handling  
- **Matplotlib / Seaborn** – visualization  
- **Scikit-learn** – preprocessing, train-test split, evaluation metrics  
- **Flask** – web deployment  
- **Pickle** – saving and loading trained models  
- **Jupyter Notebook** – interactive exploration  

---

## ⚙️ Installation & Setup  

Follow these steps to run the project locally:  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/fake-vs-real-banknote-ann.git
cd fake-vs-real-banknote-ann
