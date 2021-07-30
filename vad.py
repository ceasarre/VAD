import pandas as pd
from os.path import dirname, abspath, join
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (18,10)


FILE_PATH = "json/dataset.json"
INDEPENDENT_VAR = 'mfcc'
DEPENDED_VAR = 'label'

def load_data(file_path):

    with open(file_path, 'r') as f:
        
        data = json.load(f)
        data = json.loads(data)

    X = np.asarray(data[INDEPENDENT_VAR])
    y = np.asarray(data[DEPENDED_VAR])

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



def plot_history(history):
    
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

    fig.savefig("results\plot\plot_1.png")

def save_history(history):
    with open('results\history\history.csv', 'w') as f:
        for key in history.history.keys():
            f.write("%s,%s\n"%(key, history.history[key]))

def save_cm(model, X_test, y_test):

    # Need to choose one value
    y_pred = np.argmax(model.predict(X_test), axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(conf_matrix)
    df_cm.to_csv('results\conf_matrix\conf_matrix.csv')

    svm = sns.heatmap(conf_matrix, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
    cm, ax1 = plt.subplots(1)
    cm = svm.get_figure()    
    cm.savefig('results\conf_matrix\conf_matrix.png', dpi=400)
    

    

def main():
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

    save_history(history)
    save_cm(model, X_test, y_test)

    plot_history(history)
    
    model.save('results\model\model_1.h5')

    

if __name__ == '__main__':
    
    main()

    pass