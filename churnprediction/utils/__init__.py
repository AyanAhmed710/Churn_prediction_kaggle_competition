import os
import yaml
import sys
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger
import pickle
import numpy as np


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ChurnPredictionException(e, sys)
    

def write_yaml_file(file_path: str, data: dict):
    try:
        
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise ChurnPredictionException(e, sys)
    
def save_numpy(array : np.array, file_path: str):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ChurnPredictionException(e, sys)
    

def save_object(file_path ,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)
    except Exception as e:
        raise ChurnPredictionException(e, sys)