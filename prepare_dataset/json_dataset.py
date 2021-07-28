from os.path import dirname, abspath, join
import sys
import pandas as pd
from scipy.io import wavfile
import json

DIR_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(DIR_PATH)

DATA_PATH = "csv\Frame_descr.csv"
PATH = join(DIR_PATH,  DATA_PATH)
WAV_PATH_REL = "wav\long\data_long.wav"
WAV_PATH= join(DIR_PATH,  WAV_PATH_REL)

# Frame size in sec
FRAME_SIZE = 0.2
TARGET_FRAME_SIZE = 0.02

if __name__ == '__main__':

    # Read row data
    df = pd.read_csv(PATH)

    # drop all exept voice/unvoiced and ID
    df = df.drop(columns=[df.columns[-3],df.columns[-2], df.columns[-1]])
    df = df[df[df.columns[1]].notna()]

    df = df.astype(int)
    # remove if not 0 or 1
    df = df[df[df.columns[-1]] < 2]

    # load wav file
    SR, record = wavfile.read(WAV_PATH)

    # SET number of samples in one frame
    SAMPLES_PER_FRAME_INPUT = int(FRAME_SIZE * SR)
    SAMPLES_PER_FRAME_OUTPUT = int(TARGET_FRAME_SIZE * SR)

    data = {
        "id" : [],
        "samples" : [],
        "label" : []
    }

    i = 0
    for index, row in df.iterrows():
        print(i)
        i+=1

    pass
