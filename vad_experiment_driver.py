#----------------------------------------------------------------
# a software for testing VAD solutions with speech signal 
# containing AWGN and cocktail party noise of varying SNR
#----------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
import ast
import pandas as pd
from   tqdm import tqdm
from   scipy.io.wavfile import read
import webrtcvad
from   numpy import random
from   sklearn.preprocessing import StandardScaler
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import librosa
from   tensorflow import keras

# Lock the RNG to pre-defined seed, so results are reproducible.
random.seed(42)

# Clear the console screen (Windows/Linux).
if os.name =='nt':
    os.system('cls')
else:
    os.system('clear')

parser = argparse.ArgumentParser(description = '''A software packet capable of performing ANN VAD training and VAD algorithms accuracy evaluation.''')
parser.add_argument('-es', '--execute_stage', help='name of experimental stage to be executed, can be: ann_training, model_evaluation, or measure_train_val_acc', type=str)
args   = parser.parse_args()

#----------------------------------------------------------------
# settings and presets
#----------------------------------------------------------------

# Reading settings file from the disc.
config_h = configparser.ConfigParser()
config_h.read('settings/default.ini')

# Types discovery and conversion of the config file hadle to the
# dictionary object.
settings = {}
for section in config_h:
    settings.update({section:{}})
    for param in config_h[section]:
        settings[section].update({param:ast.literal_eval(config_h[section][param])})

# add derivative settings, so the do not have to be calculated 
# in further processing steps

settings['PROCESSING_PARAMS'].update({'samples_per_frame':int(settings['PROCESSING_PARAMS']['frame_size'] * settings['PROCESSING_PARAMS']['sampling_rate'])})

#----------------------------------------------------------------
# procedures for obtaining and converting the input data
#----------------------------------------------------------------

# A procedure which can be used to convert the JSON file to a 
# much faster-reading file in a native, binary NumPy .npz format.
def convert_json_to_npz():
    print('ładowanie danych z dysku')
    with open('dataset_50_50.json') as f:
        a = json.loads(json.load(f))

    print('zapis na dysk')
    np.savez('converted_numpy_dataset', **a)

#----------------------------------------------------------------
# procedures associated with adding noise to the audio signal
#----------------------------------------------------------------

def obtain_cocktail_party_noise_frame(frame_id, original_cocktail_noise_sgnl, frame_length, seed_increment=0):
    # Because we are manipulating the state of a random generator and 
    # we don't want  for our procedures to interfere with its operation,
    # we retrieve the current state of the RNG to restore it later.
    old_rng_state = np.random.get_state()
    
    # set the generator seed to the frame number (plus any increment)
    np.random.seed(frame_id+seed_increment)
    
    # obtain the random position of the noise frame and fetch it
    frame_start   = np.random.randint(0,len(original_cocktail_noise_sgnl)-frame_length-1)
    frame_stop    = frame_start+frame_length
    frame_samples = original_cocktail_noise_sgnl[frame_start:frame_stop]
    
    # restore the old state of the random generator
    np.random.set_state(old_rng_state)
    
    return frame_start, frame_stop, frame_samples

# signal power measurement
def measure_power(input_signal):
    return np.mean(np.power(input_signal,2))

# conversion from linear to logarithmic scale
def lin2db(power_lin):
    return 10*np.log10(power_lin)

# Function for measuring SNR.
def measure_snr(signal_clean, signal_noise):
    return lin2db(measure_power(signal_clean))-lin2db(measure_power(signal_noise))

# rms measurement
def rms_value(input_signal):
    return np.sqrt(np.mean(np.power(input_signal,2)))

# A function that adds an arbitrary noise signal  to achieve the
# desired SNR.
def add_noise_with_snr(input_signal, noise_signal, snr):
    # We modify the power of the noise so that the useful signal is 
    # by SNR decibels stronger than Gaussian noise.
    noise_signal_norm = noise_signal/rms_value(noise_signal)*rms_value(input_signal)/np.sqrt(np.power(10,snr/10))
    
    # Adding noise to the input signal.
    noisy_signal = input_signal + noise_signal_norm
    
    # We measure what SNR we actually achieved....
    acquired_snr = measure_snr(input_signal, noise_signal_norm)
    
    # ...and we return the results of our work.
    return noisy_signal, acquired_snr

def add_awgn(input_signal, snr, frame_length):
    # Generate Gaussian noise.
    awgn         = random.normal(size=frame_length)
    
    # Add noise with the specified SNR.
    noisy_signal, acquired_snr = add_noise_with_snr(input_signal, awgn, snr)
    
    return noisy_signal, acquired_snr

#----------------------------------------------------------------
# procedures associated with deep WebRTC VAD
#----------------------------------------------------------------

def calc_acc(reference_labels, classification_result):
    return 1-np.mean(np.logical_xor(reference_labels,classification_result))

def perform_webrtc_vad_tests(main_dataset, snr_level, cocktail_noise_sgnl, settings):
    def run_vads(main_dataset, snr, noise_type, cocktail_noise_sgnl, samples_per_frame):
        # Different aggressiveness settings for the VAD detector
        VADs_dict = {}
        for mode_value in settings['PROCESSING_PARAMS']['rtcvad_aggressiveness']: 
            VAD_obj = webrtcvad.Vad()
            VAD_obj.set_mode(mode_value)
            VADs_dict.update({mode_value:VAD_obj})
        
        # Test all VADs with audio frames from the original signal
        webrtc_classification_results = {}
        for aggr_val in settings['PROCESSING_PARAMS']['rtcvad_aggressiveness']:
            webrtc_classification_results.update({aggr_val:[]})
            
        pbar = tqdm(total=len(main_dataset['id']))
        for id, frame in zip(main_dataset['id'],main_dataset['samples']):
            
            if snr is not None:
                if noise_type   == 'AWGN':
                    frame,_     = add_awgn(frame,snr,samples_per_frame)
                elif noise_type == 'cocktail_party':
                    _,_,noise_frame = obtain_cocktail_party_noise_frame(id, cocktail_noise_sgnl, samples_per_frame)
                    frame,_     = add_noise_with_snr(frame, noise_frame, snr)
                else:
                    raise RuntimeError('bad noise type name was specified')
            
            for mode_value, VAD_obj in VADs_dict.items():
                webrtc_classification_results[mode_value].append(VAD_obj.is_speech(frame.astype(np.int16), settings['PROCESSING_PARAMS']['sampling_rate']))
            
            pbar.update(1)
        pbar.close()
        return webrtc_classification_results
    
    def evaluate_results(webrtc_result, main_dataset, settings, noise_type, snr):
        summary_output   = []
        reference_labels = main_dataset['label'].astype(int)
        for aggr_val in settings['PROCESSING_PARAMS']['rtcvad_aggressiveness']:
            classification_result = np.array(webrtc_result[aggr_val]).astype(int)
            acc = calc_acc(reference_labels, classification_result)
            print(f'accuracy for WebRTC VAD (aggr = {aggr_val}) is {acc}')
            summary_row = {
            'algorithm_name':'WebRTC VAD',
            'aggresiveness':aggr_val,
            'noise_type':noise_type,
            'SNR':snr,
            'accuracy':acc
            }
            summary_output.append(summary_row)
        
        return summary_output
    
    samples_per_frame = settings['PROCESSING_PARAMS']['samples_per_frame']
    
    summary_output = []
    if snr_level is None:
        print('Running WebRTC VAD algorithm (no noise)')
        webrtc_result = run_vads(main_dataset, None, None, cocktail_noise_sgnl, samples_per_frame)
        print()
        summary_output += evaluate_results(webrtc_result, main_dataset, settings, 'no_noise', None)
    else:
        print(f'Running WebRTC VAD algorithm (AWGN), SNR = {snr_level} dB')
        webrtc_result = run_vads(main_dataset, snr_level, 'AWGN', cocktail_noise_sgnl, samples_per_frame)
        print()
        summary_output += evaluate_results(webrtc_result, main_dataset, settings, 'AWGN', snr_level)
        print()
        
        print(f'Running WebRTC VAD algorithm (cocktail party), SNR = {snr_level} dB')
        webrtc_result = run_vads(main_dataset, snr_level, 'cocktail_party', cocktail_noise_sgnl, samples_per_frame)
        print()
        summary_output += evaluate_results(webrtc_result, main_dataset, settings, 'cocktail_party', snr_level)
    
    print()
    return summary_output

#----------------------------------------------------------------
# procedures associated with deep learning-based VAD
#----------------------------------------------------------------

def load_data(file_path):
    data = np.load(file_path)
    
    X = np.asarray(data['mfcc'])
    y = np.asarray(data['label'])

    # X reqired to be 4D array
    X = X[...,np.newaxis]

    return X, y

def build_model(input_shape):
    
    model = tf.keras.Sequential()

    # 1 layer
    model.add(tf.keras.layers.Conv2D(32,(2,2), activation='relu', input_shape = input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2 layer
    model.add(tf.keras.layers.Conv2D(32,(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3 layer
    model.add(tf.keras.layers.Conv2D(32,(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # TODO: LSTM?????

    # flatten output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # output
    model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

    return model

def predict(model, X, y):

    # add dimension, which is required
    
    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    # return index of the prediction
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(predicted_index, predicted_index))

def plot_history(history, settings):
    
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    fig.savefig(settings['PATHS']['trn_history_plot'])

def save_history(history, settings):
    with open(settings['PATHS']['trn_history_data'], 'w') as f:
        for key in history.history.keys():
            f.write("%s,%s\n"%(key, history.history[key]))

def save_cm(model, X_test, y_test, settings):

    # Need to choose one value
    y_pred = np.argmax(model.predict(X_test), axis=1) # model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(conf_matrix)
    df_cm.to_csv(settings['PATHS']['conf_mtx_data'])

    svm = sns.heatmap(conf_matrix, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
    cm, ax1 = plt.subplots(1)
    cm = svm.get_figure()    
    cm.savefig(settings['PATHS']['conf_mtx_plot'], dpi=400)

def ann_training(settings):

    FILE_PATH = settings['PATHS']['input_dataset']
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    X, y = load_data(FILE_PATH)
    scaler = StandardScaler()
    
    # data scaling
    #! Should the every observation be scaled or every line of the scpectrogram?
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # split for training and testing dataset. 
    #* Dataset is to big to use the KFold Cross validation on the PC
    # Spliting for train, test and validation dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # create network
    input_shape = (X_train.shape[1:])
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=100)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print('\nTest accuracy:', test_acc)
    # predict(model, X_test[1], y_test[1])

    save_history(history, settings)
    save_cm(model, X_test, y_test, settings)

    plot_history(history, settings)
    
    model.save(settings['PATHS']['ann_model_dir'])

def perform_cnnvad_vad_tests(main_dataset, cnnvad_model, snr_level, cocktail_noise_sgnl, settings):
    
    def evaluate_results(predicted_indices, reference_labels, noise_type, snr):
        acc                   = calc_acc(reference_labels, predicted_indices)
        print(f'accuracy for CNN-based VAD is {acc}')
        
        summary_row = {
        'algorithm_name':'CNN-based VAD',
        'aggresiveness':None,
        'noise_type':noise_type,
        'SNR':snr,
        'accuracy':acc
        }
        
        return summary_row
    
    def run_vads(main_dataset, cnnvad_model, snr, noise_type, cocktail_noise_sgnl, samples_per_frame):
        SR = settings['PROCESSING_PARAMS']['sampling_rate']
        
        # fetching the original parameterized dataset
        X_orig_mfcc = np.asarray(main_dataset['mfcc'])
        X_orig_mfcc = X_orig_mfcc[...,np.newaxis]
        
        # obtaining the test set (for the purpose of comparison with RTC VAD)
        y = np.asarray(main_dataset['label'])
        X_smpl = np.asarray(main_dataset['samples'])
        ids    = np.asarray(main_dataset['id'])
        
        selected_indices = np.arange(len(ids))
        
        _, X_smpl_test, _, selected_indices = train_test_split(X_smpl, selected_indices, test_size=0.2, random_state=42)
        # X_smpl_test = X_smpl
        
        sel_ids          = ids[selected_indices]
        reference_labels = y[selected_indices]
        # reference_labels = y
        
        X_mfcc_test = []
        print('parameterizing the test dataset')
        for i in tqdm(range(X_smpl_test.shape[0])):
            frame     = X_smpl_test[i,:]
            id        = sel_ids[i]
            # orig_mfcc = X_orig_mfcc[selected_indices][i,:][:,:,0]
            
            if snr is not None:
                if noise_type   == 'AWGN':
                    frame,_     = add_awgn(frame,snr,samples_per_frame)
                elif noise_type == 'cocktail_party':
                    _,_,noise_frame = obtain_cocktail_party_noise_frame(id, cocktail_noise_sgnl, samples_per_frame)
                    frame,_     = add_noise_with_snr(frame, noise_frame, snr)
                else:
                    raise RuntimeError('bad noise type name was specified')
            
            mfcc = librosa.feature.mfcc(y = frame, sr=SR, hop_length=8, n_fft=32, fmax=8000, n_mels = 10)
            X_mfcc_test.append(mfcc)
            
            # print()
            # print(orig_mfcc.shape, mfcc.shape)
            # print(orig_mfcc-mfcc)
            # exit()
            
        X_mfcc_test = np.asarray(X_mfcc_test)
        print()
        
        X_mfcc_test       = X_mfcc_test[..., np.newaxis]
        scaler            = StandardScaler()
        
        X_mfcc_test       = scaler.fit_transform(X_mfcc_test.reshape(-1, X_mfcc_test.shape[-1])).reshape(X_mfcc_test.shape)
        
        predicted_indices = np.argmax(cnnvad_model.predict(X_mfcc_test), axis=1)
        summary_row       = evaluate_results(predicted_indices, reference_labels, noise_type, snr)
        return summary_row
    
    samples_per_frame = settings['PROCESSING_PARAMS']['samples_per_frame']
    
    summary_output = []
    if snr_level is None:
        print('Running CNN-based VAD algorithm (no noise)')
        print()
        summary_output.append(run_vads(main_dataset, cnnvad_model, None, None, cocktail_noise_sgnl, samples_per_frame))
        print()
    else:
        print(f'Running CNN-based VAD algorithm (AWGN), SNR = {snr_level} dB')
        print()
        summary_output.append(run_vads(main_dataset, cnnvad_model, snr_level, 'AWGN', cocktail_noise_sgnl, samples_per_frame))
        print()
        
        print(f'Running CNN-based VAD algorithm (cocktail party), SNR = {snr_level} dB')
        print()
        summary_output.append(run_vads(main_dataset, cnnvad_model, snr_level, 'cocktail_party', cocktail_noise_sgnl, samples_per_frame))
        print()
    
    print()
    return summary_output

def measure_train_val_acc(settings, cnnvad_model):
    
    SR = settings['PROCESSING_PARAMS']['sampling_rate']
    
    # fetching the original parameterized dataset
    X_orig_mfcc = np.asarray(main_dataset['mfcc'])
    X_orig_mfcc = X_orig_mfcc[...,np.newaxis]
    
    # obtaining the test set (for the purpose of comparison with RTC VAD)
    y = np.asarray(main_dataset['label'])
    
    scaler                           = StandardScaler()
    X_orig_mfcc                      = scaler.fit_transform(X_orig_mfcc.reshape(-1, X_orig_mfcc.shape[-1])).reshape(X_orig_mfcc.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_orig_mfcc, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    predicted_indices_train = np.argmax(cnnvad_model.predict(X_train), axis=1)
    predicted_indices_val   = np.argmax(cnnvad_model.predict(X_val), axis=1)
    
    acc_train         = calc_acc(y_train, predicted_indices_train)
    acc_val           = calc_acc(y_val, predicted_indices_val)
    
    return acc_train, acc_val

#----------------------------------------------------------------
# execution of the script
#----------------------------------------------------------------

# obtain input dataset (clean audio frames)
main_dataset = np.load(settings['PATHS']['input_dataset'])

# obtain cocktail party noise source (it has to be normalized)
fs, cocktail_noise_sgnl = read(settings['PATHS']['cocktail_party_noise'])

# Sampling rate of all files is assumed to have value specified in settings file.
if fs != settings['PROCESSING_PARAMS']['sampling_rate']:
    raise RuntimeError(f"A problem with the cocktail party noise data occurred, it was assumed that the recording should have fs = {settings['PROCESSING_PARAMS']['sampling_rate']} Sa/s, but a file with a sampling rate of {fs} Sa/s was found.")

# We also assume all files to be mono!
if len(cocktail_noise_sgnl.shape) == 2:
    raise RuntimeError(f'A problem with the cocktail party noise data occurred, it was assumed that the recording is monophonic, but {cocktail_noise_sgnl.shape[1]} channels were found.')

# Normalize and convert the cocktail party noise source data.
cocktail_noise_sgnl = cocktail_noise_sgnl/np.max(np.abs(cocktail_noise_sgnl))

print()
print(f'Available main dataset keys:{list(main_dataset.keys())}')
print()

if args.execute_stage is None:
    print('Please, specify the action to be taken, read more by specifying the -h parameter.')
    print()
    
elif args.execute_stage == 'ann_training':
    ann_training(settings)
    
elif args.execute_stage == 'measure_train_val_acc':
    cnnvad_model = keras.models.load_model(settings['PATHS']['ann_model_dir'])
    acc_train, acc_val = measure_train_val_acc(settings, cnnvad_model)
    
    print()
    log_text  = ""
    log_text += "Ratios of classes in the input dataset:\n"
    for class_name in np.unique(main_dataset['label']):
        class_ratio = np.mean((main_dataset['label']==class_name).astype(int))
        class_num   = np.sum((main_dataset['label']==class_name).astype(int))
        log_text += f"\tclass \"{class_name}\": {class_ratio} ({class_num} examples)\n"
    log_text += "\n"
    
    log_text += f"Training accuracy of the model: {acc_train}, validation accuracy: {acc_val}"
    print(log_text)
    print()
    
    with open(settings['PATHS']['train_val_acc_log'], "w", encoding="utf=8") as f:
        f.write(log_text)
    
elif args.execute_stage == 'model_evaluation':
    
    if not os.path.isfile(settings['PATHS']['ann_model_dir']):
        print("The script did not find a weights data file for the CNN VAD to be loaded (checked path: {settings['PATHS']['ann_model_dir']})")
        print("Train the model first, and already then run the evaluation procedure")
        exit()
    
    cnnvad_model = keras.models.load_model(settings['PATHS']['ann_model_dir'])
    print()
    print()
    
    experiment_summary = []
    for snr_level in settings['PROCESSING_PARAMS']['snr_levels']:
        if snr_level is None:
            print('Performing experiments for clean signals.')
        else:
            print(f'Performing experiment for SNR = {snr_level} dB.')
        print()
        
        experiment_summary += perform_webrtc_vad_tests(main_dataset, snr_level, cocktail_noise_sgnl, settings)
        experiment_summary += perform_cnnvad_vad_tests(main_dataset, cnnvad_model, snr_level, cocktail_noise_sgnl, settings)

    experiment_summary = pd.DataFrame(experiment_summary)
    experiment_summary.to_excel(settings['PATHS']['experiment_summary'], index=None)
    
else:
    print('Unrecognized action name was specified.')