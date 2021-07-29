import pandas as pd
from os.path import dirname, abspath, join
import numpy as np
import json

FILE_PATH = "json/dataset_6.json"
INDEPENDENT_VAR = 'mel'
DEPENDED_VAR = 'label'

def load_data(file_path):

    with open(file_path, 'r') as f:
        
        data = json.load(f)
        data = json.loads(data)

    X = np.asarray(data[INDEPENDENT_VAR])
    y = np.asarray(data[DEPENDED_VAR])

    return X, y

        



if __name__ == '__main__':
    
    X, y = load_data(FILE_PATH)

    pass
