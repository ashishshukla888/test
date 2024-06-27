import pandas as pd
import numpy as np

import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

clf = pickle.load(open('model.pkl', 'rb'))

# Load test data
test_data = pd.read_csv('./data/feature/test_feature.csv')

# Separate features and target variable
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Ensure X_test is treated as a DataFrame and select only numerical columns
X_test_numerical = X_test.select_dtypes(include=['number'])

# Convert to NumPy array if necessary
X_test_numerical = X_test_numerical.values

# Make predictions
y_pred = clf.predict(X_test_numerical)

# Evaluate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metric_dict = {
    'mae': mae,
    'mse': mse,
    'r2': r2
}

# Save metrics to a JSON file
with open("metric.json", 'w') as file:
    json.dump(metric_dict, file, indent=4)