# Housing Price Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting housing prices using machine learning techniques.  
The goal is to build an end-to-end ML pipeline that includes data preprocessing, model training, evaluation, and comparison of multiple regression models.

The project is implemented using **scikit-learn** and follows good ML engineering practices such as pipelines, cross-validation, and proper train-test splitting.

---

## 📊 Dataset
- Dataset: Housing dataset (CSV)
- Target variable: `median_house_value`
- Features include numerical and categorical attributes such as income, location, and housing characteristics.

---

## ⚙️ Data Preprocessing
The following preprocessing steps are applied:
- Handling missing values using **median imputation**
- Feature scaling using **StandardScaler**
- Encoding categorical variables using **OneHotEncoder**
- Stratified train-test split based on income categories

All preprocessing is handled using **scikit-learn Pipelines and ColumnTransformer**.

---

## 🧠 Models Implemented
The following models were trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance is evaluated using **Root Mean Squared Error (RMSE)** with cross-validation.

---

## 📈 Evaluation Metric
- **RMSE (Root Mean Squared Error)** is used to compare model performance.
- Cross-validation ensures reliable and unbiased evaluation.

---

## 📁 Project Structure
project1/
├── data_preprocessing.ipynb   # Data cleaning and feature engineering
├── train_and_evaluate.py      # Model training and evaluation
├── model.py                   # Final training and model saving
├── .gitignore                 # Ignore model artifacts and outputs
