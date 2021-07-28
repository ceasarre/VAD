from os.path import dirname, abspath, join
import sys
from numpy.core import records
import pandas as pd
from pandas.core import frame
from scipy.io import wavfile
import json
import numpy as np

DIR_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(DIR_PATH)

DATA_PATH = "csv\Frame_descr.csv"
PATH = join(DIR_PATH,  DATA_PATH)
WAV_PATH_REL = "wav\long\data_long_mono.wav"
WAV_PATH= join(DIR_PATH,  WAV_PATH_REL)

# Frame size in sec
FRAME_SIZE = 0.2
TARGET_FRAME_SIZE = 0.02
SMALER_FRAMES = 10

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

    frame_id = 0
    for index, row in df.iterrows():
    
        frame_large = record[row[0] * SAMPLES_PER_FRAME_INPUT : (row[0] + 1) * SAMPLES_PER_FRAME_INPUT]
        label = row[1]

        for no_of_frame in range(SMALER_FRAMES):
            
            frame_small = frame_large[no_of_frame * SAMPLES_PER_FRAME_OUTPUT : (no_of_frame + 1) * SAMPLES_PER_FRAME_OUTPUT]
            data['id'].append(frame_id)
            data["samples"].append(frame_small)
            data['label'].append(label)

            frame_id += 1

    pass
