from os.path import dirname, abspath, join
import sys
import pandas as pd

DIR_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(DIR_PATH)

DATA_PATH = "csv\Frame_descr.csv"
PATH = join(DIR_PATH,  DATA_PATH)

if __name__ == '__main__':

    # Read row data
    df = pd.read_csv(PATH)

    # drop all exept voice/unvoiced and ID
    df = df.drop(columns=[df.columns[-3],df.columns[-2], df.columns[-1]])
    df = df[df[df.columns[1]].notna()]
    pass
