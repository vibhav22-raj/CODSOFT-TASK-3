import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

print("Loading the dataset...")
df = pd.read_csv("Churn_Modelling.csv")

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling numerical features...")
scaler = StandardScaler()
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
best_model = None
best_score = 0.0

print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) 
    
    results[name] = {'accuracy': accuracy, 'f1_score': f1, 'model': model}
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

print("\n--- Hyperparameter Tuning for Random Forest ---")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

tuned_rf_model = rf_grid_search.best_estimator_
y_pred_tuned_rf = tuned_rf_model.predict(X_test)
accuracy_tuned_rf = accuracy_score(y_test, y_pred_tuned_rf)
f1_tuned_rf = f1_score(y_test, y_pred_tuned_rf)

results['Tuned Random Forest'] = {'accuracy': accuracy_tuned_rf, 'f1_score': f1_tuned_rf, 'model': tuned_rf_model}

print("\nTuned Random Forest Test Results:")
print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Accuracy: {accuracy_tuned_rf:.4f}")
print(f"F1 Score: {f1_tuned_rf:.4f}")
print(classification_report(y_test, y_pred_tuned_rf))

print("\n--- Final Model Selection ---")
best_f1_score = 0.0
best_model_name = ""
for name, res in results.items():
    if res['f1_score'] > best_f1_score:
        best_f1_score = res['f1_score']
        best_model_name = name
        best_model = res['model']

print(f"The best model for this project is '{best_model_name}' with an F1 Score of {best_f1_score:.4f}")

joblib.dump(best_model, "final_churn_model.pkl")
print("\nBest model saved as 'final_churn_model.pkl'")

if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
    print(f"\nTop Features for the best model ({best_model_name}):")
    importance = best_model.feature_importances_
    features = X.columns
    for i in np.argsort(importance)[-5:][::-1]:
        print(f"{features[i]}: {importance[i]:.4f}")