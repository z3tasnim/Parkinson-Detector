# 🧠 Parkinson's Disease Detection System

This is a Python-based machine learning GUI application that predicts whether a person has Parkinson's Disease based on voice measurements. It includes features for training models, loading sample data, predicting from custom inputs or a test file, and visualizing patient data.

## 🚀 Features

- Train machine learning models (Logistic Regression, KNN, Decision Tree)
- Load and autofill sample voice data
- Make predictions using ensemble voting
- Visualize patient data as a feature bar graph
- Upload `.data` or `.csv` test files for batch predictions
- Clear input fields and reset status
- Simple and intuitive Tkinter-based GUI

## 📁 Dataset

The primary dataset used is:
- [`parkinson_disease.csv`](https://www.kaggle.com/datasets/karthickveerakumar/parkinsons-disease-detection)

Ensure this file is in the same directory as your script.

For batch predictions, you can use `.csv` or `.data` files (like the original UCI `.data` file).

## 📷 Screenshots

![Screenshot 2025-06-22 205247](https://github.com/user-attachments/assets/e65aecff-c029-4b22-92df-8b9354d0a553)

## 🧠 Features Used (22 Voice Features)

```text
'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
'spread1', 'spread2', 'D2', 'PPE'
