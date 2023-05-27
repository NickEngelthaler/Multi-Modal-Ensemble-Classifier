Python 3.10.4 (v3.10.4:9d38120e33, Mar 23 2022, 17:29:05) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Setting the working directory
os.chdir('C:\\Users\\nicke\\OneDrive\\Spring 2023\\Data Mining_\\Project\\V2')
print(os.getcwd())

# Loading the dataset
data = pd.read_csv("updated_DSData_revised_pre_ignore.csv")
X = data.drop(["greater_130k", "ID"], axis=1) # features
y = data["greater_130k"] # target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initializing the models
rcf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42)
xgb = XGBClassifier(random_state=42)

# Defining the hyperparameter spaces
rcf_param_space = {"bootstrap": [True],
        "max_depth": [6, 8, 10, 12, 14],
        "max_features": ['auto', 'sqrt','log2'],
        "min_samples_leaf": [2, 3, 4],
        "min_samples_split": [2, 3, 4, 5],
        "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # add parameter descriptions here, preloaded are default for 12000
}

logreg_param_space = {'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    # add parameter descriptions here, preloaded are default for 12000
}

xgb_param_space = {
    'n_estimators': [50, 100, 200],  # Reduced the number of values for 'n_estimators'
    'learning_rate': np.logspace(-3, 0, 4),
    'max_depth': [4, 6, 8],  # Reduced the number of values for 'max_depth'
    'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.5, 0.75, 1],
    'gamma': [0, 0.25, 0.5, 1],
    'reg_lambda': [0.1, 1, 10, 100]
    # add parameter descriptions here, preloaded are default for 12000
}

# Setting up randomized search cross-validation for hyperparameter tuning
rcf_best_params = RandomizedSearchCV(rcf, rcf_param_space, n_iter=32,
                                     scoring="roc_auc", verbose=2, cv=5,
                                     n_jobs=-1, random_state=42)
logreg_best_params = RandomizedSearchCV(lr, logreg_param_space, n_iter=160,
                                        scoring="roc_auc", verbose=2, cv=5,
                                        n_jobs=-1)
xgb_best_params = RandomizedSearchCV(xgb, xgb_param_space, n_iter=8,
                                     verbose=2, cv=3, scoring="roc_auc",
                                     n_jobs=-1)

# Fitting the models
models = [xgb_best_params, logreg_best_params, rcf_best_params]
for model in models:
    model.fit(X_train, y_train)

# Printing the best parameters and scores for each model
for model in models:
    print(model.best_params_)
    print(model.best_estimator_)
    print(model.best_score_)

# Making predictions on the validation set and evaluating the ensemble model
val_predictions = [model.predict_proba(X_val)[:, 1] for model in models]
ensemble_val_preds = np.mean(np.column_stack(val_predictions), axis=1)
roc_auc = roc_auc_score(y_val, ensemble_val_preds)
cm = confusion_matrix(y_val, (ensemble_val_preds > 0.5).astype(int))
print(f"Ensemble ROC AUC score: {roc_auc:.4f}")
print(f"Ensemble confusion matrix:\n{cm}")

# Making predictions on the test set and evaluating the ensemble model
test_predictions = [model.predict_proba(X_test)[:, 1] for model in models]
ensemble_test_preds = np.mean(np.column_stack(test_predictions), axis=1)
roc_auc = roc_auc_score(y_test, ensemble_test_preds)
cm = confusion_matrix(y_test, (ensemble_test_preds > 0.5).astype(int))
normalized_cm = cm / np.sum(cm)
print(f"Ensemble ROC AUC score: {roc_auc:.4f}")
print(f"Ensemble confusion matrix:\n{cm}")
print("Normalized confusion matrix:")
print(normalized_cm)
