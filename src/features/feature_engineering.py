import numpy as np
import pandas as pd

import os 

# 1.> feath the data 

train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# 2.> Do feature engineering

print("craete new feature here")

train_df = train_data
test_df = test_data    

# 3.> store data inside data/feature

data_path = os.path.join("data","feature")
os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_feature.csv"))
test_df.to_csv(os.path.join(data_path,"test_feature.csv"))