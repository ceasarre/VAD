#----------------------------------------------------------
# Adding Gaussian noise with set SNR 
#----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, abspath, join
from numpy import random
from scipy.io import wavfile
import sys

# Clear console
if os.name =='nt':
    os.system('cls')
else:
    os.system('clear')

# Define constant seed in RNG
random.seed(42)

#----------------------------------------------------------
# Generate "clean" signal (SNR = +infinity)
#----------------------------------------------------------

# Load input signal

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

LONG_WAV_PATH = "data_long_mono_50_50.wav"
OUTPUT_DIR = "wav\long"

# Frame size in sec
FRAME_SIZE = 0.02

if __name__ == '__main__':

# load data
    INPUT_PATH = join(d,LONG_WAV_PATH)
    OUTPUT_PATH = join(d,OUTPUT_DIR)

    SR, data = wavfile.read('data_long_mono_50_50.wav')

# number of samples in a signal
N_signal  = len(data)
t_vector  = np.linspace(0,5,N_signal)

#----------------------------------------------------------
# Measure signal power and SNR
#----------------------------------------------------------

def measure_power(input_signal):
    return np.mean(np.power(input_signal,2))

def lin2db(power_lin):
    return 10*np.log10(power_lin)

def measure_snr(signal_clean, signal_noise):
    return lin2db(measure_power(signal_clean))-lin2db(measure_power(signal_noise))

#----------------------------------------------------------
# Generate signal with given SNR
#----------------------------------------------------------

# calculate RMS of a signal
def rms_value(input_signal):
    return np.sqrt(np.mean(np.power(input_signal,2)))

# Apply additive gaussian noise to signal
def add_awgn(input_signal, snr):
    # Generate Gaussian noise
    awgn         = random.normal(size=N_signal)
    
    # Modify noise signal so that the clean signal power is SNR decibels greater than that of the noise
    awgn         = awgn/rms_value(awgn) * rms_value(input_signal)/np.sqrt(np.power(10,snr/10))

    # Add noise to input signal
    noisy_signal =  input_signal + awgn

    # Measure SNR of the resulting signal
    acquired_snr = measure_snr(input_signal, awgn)
    
    return noisy_signal, acquired_snr

snr_vec = [-5,0,5,10,15]

# Save signals with varying levels of SNR
for i, snr in enumerate(snr_vec):
    noisy_signal, acquired_snr = add_awgn(data, snr)
    wavfile.write("long_awgn_SNR{}.wav".format(snr),SR,noisy_signal.astype(np.int16))