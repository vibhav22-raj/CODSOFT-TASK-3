# CODSOFT-TASK-3

# CodSoft Machine Learning Internship  

This repository contains the solutions to three machine learning projects completed during the *CodSoft Internship*. Each task applies ML techniques to solve real-world problems such as fraud detection, churn prediction, and spam classification.  

---

# Customer Churn Prediction Model

A machine learning project to predict customer churn using multiple classification algorithms with hyperparameter tuning and model comparison.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [File Structure](#file-structure)

## 🎯 Overview

This project implements a comprehensive customer churn prediction system that:
- Processes and prepares customer data for machine learning
- Trains multiple classification models (Logistic Regression, Random Forest, Gradient Boosting)
- Performs hyperparameter tuning for optimal performance
- Compares models based on F1-score and accuracy
- Saves the best performing model for deployment

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset with the following characteristics:
- **Target Variable**: `Exited` (1 = churned, 0 = retained)
- **Features**: Customer demographics, account information, and transaction data
- **Preprocessing**: Removes unnecessary columns (`RowNumber`, `CustomerId`, `Surname`)
- **Encoding**: One-hot encoding for categorical variables (`Geography`, `Gender`)

### Key Features Used:
- `CreditScore`: Customer's credit score
- `Age`: Customer age
- `Tenure`: Years with the bank
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used
- `HasCrCard`: Has credit card (binary)
- `IsActiveMember`: Active membership status (binary)
- `EstimatedSalary`: Estimated salary
- `Geography_*`: Country indicators (one-hot encoded)
- `Gender_*`: Gender indicator (one-hot encoded)

## 🛠️ Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
```

## 📥 Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd churn-prediction
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn joblib
```

3. Ensure you have the dataset `Churn_Modelling.csv` in the same directory as the script.

## 🚀 Usage

Run the churn prediction model:

```bash
python churn_prediction.py
```

The script will:
1. Load and preprocess the dataset
2. Train multiple machine learning models
3. Perform hyperparameter tuning
4. Display model performance metrics
5. Save the best model as `final_churn_model.pkl`

## 📈 Model Performance

The script evaluates three models:

1. **Logistic Regression**: Baseline linear model with class balancing
2. **Random Forest**: Ensemble method with class balancing
3. **Gradient Boosting**: Advanced ensemble method
4. **Tuned Random Forest**: Optimized Random Forest with grid search

### Evaluation Metrics:
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Classification Report**: Detailed precision, recall, and F1-score per class

## 🔧 Technical Details

### Data Preprocessing:
- Removes irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Train-test split (80-20)

### Hyperparameter Grid Search:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}
```

### Model Selection:
- Models are compared based on F1-score
- Best model is automatically saved
- Feature importance is displayed for tree-based models

## 📁 File Structure

```
project/
│
├── churn_prediction.py     
├── Churn_Modelling.csv      
├── final_churn_model.pkl   
└── README.md               
```

## 💡 Key Features

- **Automated Pipeline**: Complete ML pipeline from data loading to model saving
- **Class Imbalance Handling**: Uses balanced class weights
- **Feature Engineering**: Proper encoding and scaling
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Hyperparameter Optimization**: Grid search for best performance
- **Feature Importance**: Identifies most predictive features
- **Production Ready**: Saves trained model for deployment

## 🎯 Expected Output

The script will display:
- Model training progress
- Performance metrics for each model
- Hyperparameter tuning results
- Best model selection
- Top 5 most important features
- Saved model confirmation

## 🔮 Future Enhancements

- Add more advanced models (XGBoost, Neural Networks)
- Implement cross-validation
- Add feature selection techniques
- Create prediction API
- Add data visualization
- Implement model monitoring

## 📄 License

This project is open source and available under the MIT License.
