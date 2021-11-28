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

DATA_PATH = "csv\\5_speakers.csv"
PATH = join(DIR_PATH,  DATA_PATH)
JSON_FILE_PATH = join(DIR_PATH, "json\dataset_speaker_5.json")

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

    # drop all exept voice\\unvoiced and ID
    # df = df.drop(columns=[df.columns[-3],df.columns[-2], df.columns[-1]])
    df.head()   # testing if your object has the right type of data
    df = df.iloc[:, 1:3]
    df = df[df[df.columns[1]].notna()]  # zostawia tylko wpisane wartosci

    data = {    # tablica list
        "id": [],
        "samples": [],
        "mel": [],
        "mfcc": [],
        "label": []
    }

    frame_id = 0
    for index, row in df.iterrows():
        WAV_PATH = row[0]
        # WAV_PATH = "wavs\\speakers\\2BqVo8kVB2Skwgyb\\0a3129c0-4474-11e9-a9a5-5dbec3b8816a.wav"
        WAV_PATH = join(DIR_PATH,  WAV_PATH)
        SR, record = wavfile.read(WAV_PATH)
        label = row[1]  # get speaker ID, which will be later appended to a list
        SAMPLES_PER_FRAME = int(TARGET_FRAME_SIZE * SR)
        frames_in_file = len(record) // (SAMPLES_PER_FRAME)
        for file_frame_counter in range(frames_in_file):
            frame = record[file_frame_counter * SAMPLES_PER_FRAME: (file_frame_counter + 1) * SAMPLES_PER_FRAME]
            frame = np.asarray(frame, dtype="float64")
            # ! DO Konsultacji
            melscpect = librosa.feature.melspectrogram(
                y=frame, sr=SR, hop_length=8, n_fft=32, fmax=8000, n_mels=10)
            mfcc = librosa.feature.mfcc(
                y=frame, sr=SR, hop_length=8, n_fft=32, fmax=8000, n_mels=10)

            data['id'].append(frame_id)
            data["samples"].append(frame)
            data['label'].append(label)
            data['mel'].append(melscpect)
            data['mfcc'].append(mfcc)
            file_frame_counter += 1
            frame_id += 1
            print("ID NR {}".format(frame_id))
        # wyciagnieto wszystkie ramki z pliku, licznik zerowany
        file_frame_counter = 0

    # ! WHEN READING JSON FILE, TO USE IT, CONVERT TO NP ARRAY
    # data['id'] = np.asarray(data['id'])
    # data['samples'] = np.asarray(data['samples'])
    # data['label'] = np.asarray(data['label'])

    # dump file

    dumped = json.dumps(data, cls=NumpyEncoder)  # zakodowanie danych do jsona

    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump(dumped, json_file)

    print('---------------------------- zapisano w {} --------------------------'.format(JSON_FILE_PATH))
