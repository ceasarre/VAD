#----------------------------------------------------------
# Adding cocktail party noise with set SNR
#----------------------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from os.path import dirname, abspath, join
import sys

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

# Assume frame length of 20 ms.
FRAME_LENGTH = 0.020 # [s]

# Clear console
if os.name=='nt':
    os.system('cls')
else:
    os.system('clear')

#----------------------------------------------------------
# Load Data
#----------------------------------------------------------

# Load noise frames source
fs, original_cocktail_noise_sgnl = read('23153__freqman__party-sounds_16kHzMono.wav')

# Calculate frame length in samples
FRAME_LENGTH_SMPL  = int(fs*FRAME_LENGTH)

# Calculate number of noise frames
noise_frames = len(original_cocktail_noise_sgnl) / FRAME_LENGTH_SMPL

# normalize and convert to float
original_cocktail_noise_sgnl = original_cocktail_noise_sgnl/np.max(np.abs(original_cocktail_noise_sgnl))

#----------------------------------------------------------
# Cocktail party noise overlay procedures
#----------------------------------------------------------

def obtain_cocktail_noise_frame(frame_id, original_cocktail_noise_sgnl, frame_length, seed_increment=0):
    # acquire current RNG state
    old_rng_state = np.random.get_state()
    
    # set seed to frame id (plus increment)
    np.random.seed(frame_id+seed_increment)
    
    # randomize noise frame timestamp and load it
    frame_start   = np.random.randint(0,len(original_cocktail_noise_sgnl)-frame_length-1)
    frame_stop    = frame_start+frame_length
    frame_samples = original_cocktail_noise_sgnl[frame_start:frame_stop]
    
    # return old RNG state
    np.random.set_state(old_rng_state)
    
    return frame_start, frame_stop, frame_samples

# measure signal power
def measure_power(input_signal):
    return np.mean(np.power(input_signal,2))

# calc linear scale to logarithmic
def lin2db(power_lin):
    if (power_lin == 0 ): 
        power_lin = 1
    return 10*np.log10(power_lin)

def measure_snr(signal_clean, signal_noise):
    return lin2db(measure_power(signal_clean))-lin2db(measure_power(signal_noise))

# measure RMS
def rms_value(input_signal):
    return np.sqrt(np.mean(np.power(input_signal,2)))

# Add noise signal in a way that ensures given SNR value 
def add_noise_with_snr(input_signal, noise_signal, snr):
    # Modify noise signal so that the clean signal power is SNR decibels greater than that of the gaussian noise
    noise_signal_norm = noise_signal/rms_value(noise_signal)*rms_value(input_signal)/np.sqrt(np.power(10,snr/10))
    # Add noise signal to input signal
    noisy_signal = input_signal + noise_signal_norm
    
    # Measure actual SNR of acquired signal
    acquired_snr = measure_snr(input_signal, noise_signal_norm)
    
    return noisy_signal, acquired_snr

def get_data_frame(frame_id):
    data_frame = data[int(frame_id * FRAME_LENGTH_SMPL):int(frame_id * FRAME_LENGTH_SMPL + FRAME_LENGTH_SMPL)]
    return data_frame
#----------------------------------------------------------
# Implementation
#----------------------------------------------------------

# Load input signal
LONG_WAV_PATH = "wav\long\data_long_mono_50_50.wav"
OUTPUT_DIR = "wav\long"

INPUT_PATH = join(d,LONG_WAV_PATH)
OUTPUT_PATH = join(d,OUTPUT_DIR)
SR, data = read("short.wav")

# Calculate number of frames = number of samples / (number of samples per frame)
frames = int(len(data) / (FRAME_LENGTH * SR))

# Signal will be mixed with noise with different SNR values, ranging from -5 dB to 15 dB
snr_vec = [-5,0,5,10,15]
full_data_array = []
contaminated_data_SNR = []

# Loop over every given SNR, add noise and save the result
for i, snr in enumerate(snr_vec):
    for id in range(frames):
        _,_,noise_frame = obtain_cocktail_noise_frame(id, original_cocktail_noise_sgnl, FRAME_LENGTH_SMPL)
        data_frame = get_data_frame(id)
        contaminated_data, acquired_snr = add_noise_with_snr(data_frame, noise_frame, snr)
        full_data_array.extend(contaminated_data)
    wavfile.write('contaminated_signal_SNR{}.wav'.format(snr),SR,np.array(full_data_array).astype(np.int16))
    full_data_array = []