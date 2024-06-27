import numpy as np
import pandas as pd
import os

from scipy import stats                 
from sklearn import linear_model        
from sklearn import preprocessing       
from sklearn import model_selection     
from sklearn import tree                
from sklearn import ensemble            
from sklearn import metrics             
from sklearn import cluster             
from sklearn import feature_selection   

import warnings
warnings.filterwarnings('ignore')


# 1.> fetch the data from data/raw

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

# 2.> transform the data

print("Preprocessed data")

train_processed_data = train_data
test_processed_data = test_data

# 3.> store the data inside data/processed 
 
data_path = os.path.join("data","processed")
os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))

# Additional Suggestions
# Backtesting: Implement a backtesting framework to evaluate the performance of your strategy using historical data.
# Risk Management: Incorporate risk management techniques such as stop-loss and take-profit mechanisms.
# Portfolio Optimization: Use algorithms to optimize your portfolio allocation.
# Sentiment Analysis: Integrate news sentiment analysis to add another layer of decision-making.
# Visualization: Visualize your data, model predictions, and decision-making process using libraries like matplotlib, seaborn, 
# and Plotly.