from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load dataset
proteins_data = pd.read_csv("./Proj Train/train_proteins.csv")
peptides_data = pd.read_csv("./Proj Train/train_peptides.csv")
clinical_data = pd.read_csv("./sample1.csv")

# Aggregating data
aggregated_data = clinical_data.groupby('patient_id').agg({
    'visit_id': 'first',
    'visit_month': 'sum',
    'updrs_1': 'mean',
    'updrs_2': 'mean',
    'updrs_3': 'mean',
    'upd23b_clinical_state_on_medication': 'first'
}).reset_index()

# Train-Test Split
X = aggregated_data.drop(['updrs_1', 'updrs_2', 'updrs_3'], axis=1)
y = aggregated_data[['updrs_1', 'updrs_2', 'updrs_3']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Handle NaN values in y_train
missing_rows = y_train.isnull().any(axis=1)
X_train = X_train[~missing_rows]
y_train = y_train[~missing_rows]

# Hyperparameter tuning for Random Forest Regressor
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Model Evaluation using SMAPE
def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Model Evaluation
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Best Mean Squared Error: {mse}')

predictions = best_model.predict(X_test)
smape = calculate_smape(y_test, predictions)
print(f'Best Symmetric Mean Absolute Percentage Error (SMAPE): {smape}')

# Prediction on new data
new_data = pd.read_csv("./sample1.csv")
new_data = new_data[X.columns]
new_predictions = best_model.predict(new_data)

# Print or use new_predictions as needed
print("Predictions on new data with the best model:")
print(new_predictions)