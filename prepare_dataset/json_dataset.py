from os.path import dirname, abspath, join
import sys
from numpy.core import records
import pandas as pd
from pandas.core import frame
from scipy.io import wavfile
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


DIR_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(DIR_PATH)

DATA_PATH = "csv\Frame_descr.csv"
PATH = join(DIR_PATH,  DATA_PATH)
WAV_PATH_REL = "wav\long\data_long_mono.wav"
WAV_PATH= join(DIR_PATH,  WAV_PATH_REL)
JSON_FILE_PATH = join(DIR_PATH, "json\dataset.json")

# Frame size in sec
FRAME_SIZE = 0.2
TARGET_FRAME_SIZE = 0.02
SMALER_FRAMES = 10


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
        "mel" : [],
        "mfcc": [],
        "label" : []
    }

    frame_id = 0
    for index, row in df.iterrows():
    
        frame_large = record[row[0] * SAMPLES_PER_FRAME_INPUT : (row[0] + 1) * SAMPLES_PER_FRAME_INPUT]
        label = row[1]

        # !TEST
        # frame_small = record[row[0] * SAMPLES_PER_FRAME_INPUT : (row[0] + 3) * SAMPLES_PER_FRAME_INPUT]
        # frame_small = np.asarray(frame_small, dtype="float64")
        # melscpect = librosa.feature.melspectrogram(y = frame_small, sr = SR, hop_length=32)
        # fig, ax = plt.subplots()
        # S_dB = librosa.power_to_db(melscpect, ref=np.max)
        # img = librosa.display.specshow(S_dB, x_axis='time',
        #                         y_axis='mel', sr=SR,
        #                         fmax=8000, ax=ax)
        # fig.colorbar(img, ax=ax, format='%+2.0f dB')
        # ax.set(title='Mel-frequency spectrogram')


        for no_of_frame in range(SMALER_FRAMES):
            
            frame_small = frame_large[no_of_frame * SAMPLES_PER_FRAME_OUTPUT : (no_of_frame + 1) * SAMPLES_PER_FRAME_OUTPUT]
            frame_small = np.asarray(frame_small, dtype="float64")

            # ! DO Konsultacji
            melscpect = librosa.feature.melspectrogram(y = frame_small, sr = SR, hop_length=8, n_fft=32, fmax=8000, n_mels = 10)
            mfcc = librosa.feature.mfcc(y = frame_small, sr=SR, hop_length=8, n_fft=32, fmax=8000, n_mels = 10)
            
            data['id'].append(frame_id)
            data["samples"].append(frame_small)
            data['label'].append(label)
            data['mel'].append(melscpect)
            data['mfcc'].append(mfcc)

            frame_id += 1
            print("ID NR {}".format(frame_id))

    # ! WHEN READING JSON FILE, TO USE IT, CONVERT TO NP ARRAY
    # data['id'] = np.asarray(data['id'])
    # data['samples'] = np.asarray(data['samples'])
    # data['label'] = np.asarray(data['label'])
    
    # dump file 

    dumped = json.dumps(data, cls=NumpyEncoder)

    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump(dumped, json_file)

    print('---------------------------- zapisano w {} --------------------------'.format(JSON_FILE_PATH))
