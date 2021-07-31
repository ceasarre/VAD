from tkinter import Frame
# from frame import Frame
import json
import numpy as np
from math import log10
import matplotlib.pyplot as plt

# path
DATA_PATH = "json/dataset_6.json"
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

# def wgn(x, snr):
#     snr = 10**(snr/10.0)
#     xpower = np.sum(x**2)/len(x)
#     npower = xpower / snr
#     return np.random.randn(len(x)) * np.sqrt(npower)
    
# main
if __name__ == '__main__':
    X_voice, Y_voice, x_noise, y_noise, samples, labels  = load_data(DATA_PATH)

    # t = len(samples) * 320
    # x = samples
    # n = wgn(x, 6)
    # Xn = x+n # signal with 6dBz SNR noise added
    # pl.subplot(211)
    # pl.hist(n, bins=100, normed=True)
    # pl.subplot(212)
    # pl.psd(n)
    # pl.show()

    # X_noise, y_noise = load_data(DATA_PATH, 0)
    
    energy_voice = 0
    counter = 0
    temp = 0
    for samples in X_voice:
        energy_voice += samples
        counter +=1

    energy_voice = energy_voice/counter
    
    energy_noise = 0
    counter = 0
    temp = 0
    
    for samples in x_noise:
        energy_noise += samples
        counter+=1

    energy_noise = energy_noise/counter
    SNR = 10*log10( sum(energy_voice**2) / sum(energy_noise**2))
    print(SNR)