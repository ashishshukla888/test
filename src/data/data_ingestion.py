import pandas as pd
import os
from sklearn.model_selection import train_test_split
from urllib.error import HTTPError
import yaml
import logging

# configure logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(parms_path:str) -> float:
    try:
        test_size = yaml.safe_load(open(parms_path,'r'))['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        #print(f"Error: File '{parms_path}' not found.")
        # return 0.2  # Default test_size or raise an error as needed
    except yaml.YAMLError as exc:
        logger.error('yaml File not found')
        # print(f"Error in YAML file: {exc}")
        # return 0.2  # Default test_size or raise an error as needed
    except Exception as e:
        logger.error('some error occured')
        return e
 

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except HTTPError as e:
        print(f"HTTPError: {e}")
        # return pd.DataFrame() # or can return empty data frame

# Create a Path to store datadvc 

def save_data(data_path: str,train_data: pd.DataFrame,test_data: pd.DataFrame) -> None:
    
    try:
        os.makedirs(data_path)
        train_data.to_csv(os.path.join(data_path,"train.csv"))
        test_data.to_csv(os.path.join(data_path,"test.csv"))
    except IOError as e:
        print(f"IOError: {e}")

def main():

    test_size = load_params('params1.yaml')
    df = read_data("https://raw.githubusercontent.com/TanmayKedari/Exploratory-Analysis-of-NYC-Taxi/master/taxi_zones.csv")

    train_data, test_data = train_test_split(df,test_size=test_size, random_state=42)

    data_path = os.path.join("data","raw")
    save_data(data_path,train_data,test_data)

if __name__ == "__main__":
    main()