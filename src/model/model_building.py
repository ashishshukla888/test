import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

train_data = pd.read_csv('./data/feature/train_feature.csv')
# print(train_data.head(5))

# Separate features and target variable
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Ensure X_train is treated as a DataFrame and select only numerical columns
X_numerical = X_train.select_dtypes(include=['number'])

# Convert to NumPy array
X_numerical = X_numerical.values

params = yaml.safe_load(open('params.yaml','r'))['model_building']

# Create and fit the model
clf = GradientBoostingRegressor(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
clf.fit(X_numerical, y_train)

# Save the model
pickle.dump(clf, open('model.pkl', 'wb'))