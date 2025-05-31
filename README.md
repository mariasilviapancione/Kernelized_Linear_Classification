# Kernelized Linear Classification

This project implements and compares a series of binary classification algorithms entirely from scratch, without relying on machine learning libraries such as Scikit-learn. The main goal is to evaluate model performance using the 0–1 loss metric and investigate the trade-offs between model complexity, accuracy, and computational efficiency.

## 📌 Project Overview

The workflow includes:

- **Exploratory Data Analysis & Preprocessing**  

- **Linear Models**  
  Implementation of:
  - Perceptron  
  - Support Vector Machines using the Pegasos algorithm  
  - Regularized logistic classification 

- **Polynomial Feature Expansion**  
  Use of degree-2 polynomial expansion to enable non-linear decision boundaries.

- **Kernel Methods**  
  Implementation of:
  - Kernelized Perceptron (Gaussian & Polynomial kernels)  
  - Kernelized Pegasos SVM (Gaussian & Polynomial kernels)

- **Hyperparameter Tuning**  
  Systematic search with **5-fold cross-validation** to identify optimal settings for each model.

- **Final Evaluation**  
  Comparative analysis of all models in terms of performance, generalization, and computational cost.

## 🛠 Technologies

- Language: **Python**

