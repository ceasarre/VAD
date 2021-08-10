from tkinter import Frame
import json
import numpy as np
from math import log10
import matplotlib.pyplot as plt

# path
DATA_PATH = "json/dataset_short.json"
# DATA_PATH_VOICE= r"./voice.json"

# load data
def load_data(data_path):

    X = []
    Y = []
    x = []
    y = []

    with open(data_path) as fp:
        data = json.load(fp)
        data = json.loads(data)

    samples = np.array(data["samples"])
    labels = np.array(data["label"])
    # mapping = np.array(data["mapping"])

    for label in labels:
        if label == 1:
            X.append(samples[label])
            Y.append(labels[label])
        else:
            x.append(samples[label])
            y.append(labels[label])
    
    return X,Y,x,y,samples,labels 

# measure signal power
def measure_power(input_signal):
    return np.mean(np.power(input_signal,2))

# calc linear scale to logarithmic
def lin2db(power_lin):
    return 10*np.log10(power_lin)

def measure_snr(signal_clean, signal_noise):
    return lin2db(measure_power(signal_clean))-lin2db(measure_power(signal_noise))
    
# main
if __name__ == '__main__':
    X_voice, Y_voice, x_noise, y_noise, samples, labels  = load_data(DATA_PATH)
    voice_signal = []
    noise_signal = []
    for i in range(len(X_voice)):
        voice_signal.extend(X_voice[i])
    for i in range(len(x_noise)):
        noise_signal.extend(x_noise[i])    
    snr = measure_snr(voice_signal, noise_signal)