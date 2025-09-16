This repository contains the full implementation and dataset for a Bachelor’s thesis project titled:

"Developing a Machine Learning-Based Decision Support System for Food Health Risk Assessment through the NOVA Classification"

📌 Project Overview

This study presents the design, development, and implementation of a machine learning–based Decision Support System (DSS) capable of assessing the health risks of food products based on their ingredient composition. The system leverages the NOVA classification, a globally recognized framework that categorizes food based on its degree of industrial processing.

Users can input ingredient lists, and the system predicts:

NOVA Group (1–4)

Mapped Health Risk Level (Low / Moderate / High)

Presence of health-risk indicators including:

Carcinogens

Allergens

Cardiovascular-risk compounds

The DSS is implemented with a Streamlit GUI and trained on a manually curated dataset of 200 real frozen food products sourced from Open Food Facts
.

📁 Repository Contents

File/Folder	Description

Final Dataset.csv            Cleaned and structured dataset used for model training and evaluation.

Final Database.xlsx	         Original raw database (manually curated) with ingredient and food product info.

builtataset.py	             Script for preprocessing and transforming raw food data into model-ready features.

logistic regression.py	     Logistic Regression implementation with training, evaluation, and metrics.

decisiontree.py	             Decision Tree model implementation and evaluation.

neuralnetwork.py	           Neural Network model implementation (Keras/TensorFlow).

final_logreg_model.py	       Final version of the trained logistic regression model script.

final_logreg_model.joblib	   Saved logistic regression model (used by the DSS).

dss.py	                     Streamlit app that allows ingredient list input and displays NOVA prediction + risk factors.

📊 Machine Learning Models

Three models were trained and compared:

Logistic Regression (selected for final deployment)

Decision Tree

Neural Network (Feedforward MLP)

Evaluation:
All models were trained using 10-fold cross-validation and evaluated on Accuracy, Precision, Recall, and F1-score.

Best model: Logistic Regression (95% accuracy, consistent generalization)

🧠 NOVA Classification

The system classifies food into the following NOVA groups:

Group 1: Unprocessed or minimally processed foods

Group 2: Processed culinary ingredients

Group 3: Processed foods

Group 4: Ultra-processed foods (UPFs)

Prediction is based on structured features extracted from ingredient lists, including additive counts, ingredient ratios, and known health-risk indicators (carcinogens, allergens, etc.).

📌 Use Cases

Consumers looking to make healthier food choices

Public health authorities and researchers studying dietary risks

Nutritionists promoting reduced UPF intake

Regulatory bodies needing automated food classification

📚 License and Attribution

This codebase is intended for educational and research purposes.

🙏 Acknowledgments

This project was developed as part of a Bachelor's thesis at the Faculty of Economics, University of Tirana.

Supervised by: Assoc. Prof. Dr. Majlinda Godolja

Dataset sourced from: Open Food Facts
