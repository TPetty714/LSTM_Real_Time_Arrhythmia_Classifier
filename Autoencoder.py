import time
from sklearn.model_selection import train_test_split as Split
from sklearn.preprocessing import StandardScaler
import os
import sys
from os import path
import wfdb
from wfdb import rdrecord, rdann
import numpy as np
import more_itertools as mit
import pickle
from matplotlib import pyplot as plt
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from biosppy.signals import ecg
from keras.utils.vis_utils import plot_model
import scipy.signal
from sklearn.preprocessing import normalize

EPOCHS = 50
VERBOSE = 1
WINDOW_SIZE = 360

CLASSIFICATION = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                   'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')
]


def get_data_from_csv(filename):
    df = pd.read_csv(filename)
    #if filename == "training_by_classes.csv" or filename == "validating_by_classes.csv":
    #    df = df[df.iloc[:,1]=='N']
        #print(df.head())
    #X = df.iloc[:, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
    X = df.iloc[:, 360-180: 360+180]
    annotations = df.iloc[:, 1]
    y = np.array(annotations).reshape((-1, 1))
    X = X.to_numpy().reshape(-1, WINDOW_SIZE, 1)
    nyquist = 360/2
    high = 40/nyquist
    low = 0.5/nyquist
    #sos = scipy.signal.butter(5, [low, high], "band", output='sos')
    b, a = scipy.signal.butter(5, [low, high], "band")
    for x in range(X.shape[0]):
        #signals2 = scipy.signal.sosfilt(sos, X[x].flatten())
        signals_butter = scipy.signal.filtfilt(b, a, X[x].flatten())
        signals = signals_butter.reshape(WINDOW_SIZE, 1)
        signals = normalize(signals, axis=0)
        plt.plot(signals, label="Augmented")
        plt.plot(X[x].flatten(), label="Original")
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        #plt.plot(signals2, label="SOS Filtered")
        
        plt.legend()
        plt.title(y[x][0])
        plt.show()
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)        
    return X, y


def create_model():
    encoder_input = keras.Input(shape=(WINDOW_SIZE, 1), name='input')
    #x = keras.layers.Dense(WINDOW_SIZE, activation='tanh')(encoder_input)
    x = keras.layers.LSTM(90, return_sequences=True)(encoder_input)
    x = keras.layers.LSTM(45, return_sequences=False)(x)
    encoder_output = keras.layers.Reshape((45,1))(x)

    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    #encoder.summary()

    decoder_input = keras.layers.Input(shape=(45, 1), name='decoder_input')
    x = keras.layers.LSTM(45, return_sequences=True)(decoder_input)
    x = keras.layers.LSTM(90, return_sequences=False)(x)
    x = keras.layers.Dense(WINDOW_SIZE, activation='tanh')(x)
    decoder_output = keras.layers.Reshape((WINDOW_SIZE, 1))(x)

    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    #decoder.summary()

    opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')
    #autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer=opt)
    return autoencoder, encoder, decoder


def train_on_records(local_dir, training, validating, model):
    X, y = get_data_from_csv(training)
    vx, vy = get_data_from_csv(validating)
    history = model.fit(X, X, epochs=EPOCHS, verbose=VERBOSE, shuffle=True, validation_data=(vx, vx))

    plt.plot(history.history['loss'])
    plt.title("Losses")
    plt.show()
    return model
   

def evaluate_on_records(model, records):
    X, y = get_data_from_csv(records)
    X_test_pred = model.predict(X)
    Loss = np.abs(np.subtract(X, X_test_pred)).mean(axis=1)
    threshold = np.mean(Loss) + np.std(Loss)
    return threshold

def predict_on_records(model, records, threshold):
    sum = 0
    count = 0
    start = time.time()
    yp = np.array([[]])
    df = pd.read_csv(records)
    #df = df.iloc[:10, :]
    X = df.iloc[:, 360-int(WINDOW_SIZE/2):360+int(WINDOW_SIZE/2)]
    X = X.to_numpy().reshape(-1, WINDOW_SIZE, 1)
    nyquist = 360/2
    high = 40/nyquist
    low = 0.5/nyquist
    b, a = scipy.signal.butter(5, [low, high], "band")
    for x in range(X.shape[0]):
        signals_butter = scipy.signal.filtfilt(b, a, X[x].flatten())
        signals = signals_butter.reshape(WINDOW_SIZE, 1)
        signals = normalize(signals, axis=0)
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        if yp.shape[1] == 0:
            yp = model.predict(X[x].reshape((1, WINDOW_SIZE, 1)))
        else:
            yp = np.append(yp, model.predict(X[x].reshape((1, WINDOW_SIZE, 1))), axis=0)
        current = time.time()
        #print(current-start)
        sum += current-start
        count += 1
        start = time.time()
    X_test_pred = yp
    print("Average run time for each record:", sum/count)
    #X, y = get_data_from_csv(records)
    #X_test_pred.append(model.predict(X, verbose=1))

    annotations = df.iloc[:, 1]
    y = np.array(annotations).reshape((-1, 1))

    figure, axis = plt.subplots(3, 3)

    for i in range(len(y)):
        if y[i] == 'N':
            axis[0,0].plot(X[i], label='Original')
            axis[0,0].plot(X_test_pred[i], label='Reconstructed')
            axis[0,0].set_title(y[i])
            axis[0,0].get_xaxis().set_visible(False)
            axis[0,0].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'L':
            axis[0,1].plot(X[i], label='Original')
            axis[0,1].plot(X_test_pred[i], label='Reconstructed')
            axis[0,1].set_title(y[i])
            axis[0,1].get_xaxis().set_visible(False)
            axis[0,1].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'R':
            axis[0,2].plot(X[i], label='Original')
            axis[0,2].plot(X_test_pred[i], label='Reconstructed')
            axis[0,2].set_title(y[i])
            axis[0,2].get_xaxis().set_visible(False)
            axis[0,2].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'V':
            axis[1,0].plot(X[i], label='Original')
            axis[1,0].plot(X_test_pred[i], label='Reconstructed')
            axis[1,0].set_title(y[i])
            axis[1,0].get_xaxis().set_visible(False)
            axis[1,0].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == '/':
            axis[1,1].plot(X[i], label='Original')
            axis[1,1].plot(X_test_pred[i], label='Reconstructed')
            axis[1,1].set_title(y[i])
            axis[1,1].get_xaxis().set_visible(False)
            axis[1,1].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'A':
            axis[1,2].plot(X[i], label='Original')
            axis[1,2].plot(X_test_pred[i], label='Reconstructed')
            axis[1,2].set_title(y[i])
            axis[1,2].get_xaxis().set_visible(False)
            axis[1,2].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'F':
            axis[2,0].plot(X[i], label='Original')
            axis[2,0].plot(X_test_pred[i], label='Reconstructed')
            axis[2,0].set_title(y[i])
            axis[2,0].get_xaxis().set_visible(False)
            axis[2,0].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'f':
            axis[2,1].plot(X[i], label='Original')
            axis[2,1].plot(X_test_pred[i], label='Reconstructed')
            axis[2,1].set_title(y[i])
            axis[2,1].get_xaxis().set_visible(False)
            axis[2,1].get_yaxis().set_visible(False)
            break

    for i in range(len(y)):
        if y[i] == 'j':
            axis[2,2].plot(X[i], label='Original')
            axis[2,2].plot(X_test_pred[i], label='Reconstructed')
            axis[2,2].set_title(y[i])
            axis[2,2].get_xaxis().set_visible(False)
            axis[2,2].get_yaxis().set_visible(False)
            break

    plt.legend()
    plt.show()

    X_test_mae = np.abs(np.subtract(X, X_test_pred)).mean(axis=1)
    #X_test_mse = np.square(np.subtract(X, X_test_pred)).mean(axis=1)
    #sns.displot(X_test_mae, bins=50, kde=True)
    #plt.title("MAE")
    #plt.show()
    #sns.displot(X_test_mse, bins=50, kde=True)
    #plt.title("MSE")
    #plt.show()

    test_score_df = pd.DataFrame(index=range(len(X_test_mae)))
    test_score_df['loss'] = X_test_mae
    #test_score_df['mse'] = X_test_mse
    test_score_df['Annotation'] = y

    #annotations = ['N', 'L', 'R', 'A', 'a', 'J', 'S']
    annotations = ['N']
    normal = test_score_df.loc[test_score_df['Annotation'].isin(annotations)]
    abnormal = test_score_df.loc[~test_score_df['Annotation'].isin(annotations)]
    #threshold = np.mean(normal['loss']) + 2*np.std(normal['loss'])

    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

    #threshold2 = np.mean(normal['mse']) + np.std(normal['mse'])
    #test_score_df['threshold2'] = threshold2
    #test_score_df['anomaly2'] = test_score_df.mse > test_score_df.threshold
    
    print(test_score_df.head())

    plt.boxplot(test_score_df["loss"])
    plt.title("MAE")
    plt.show()
    #plt.boxplot(test_score_df["mse"])
    #plt.title("MSE")
    #plt.show()


    #plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    #plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    #plt.xticks(rotation=25)
    #plt.legend()
    #plt.show()

    print("Normal mae mean:", np.mean(normal['loss']), " std:", np.std(normal['loss']))
    #print("Normal mse mean:", np.mean(normal['mse']), " std:", np.std(normal['mse']))

    print("Abnormal mean:", np.mean(abnormal['loss']), " std:", np.std(abnormal['loss']))
    #print("Abnormal mse mean:", np.mean(abnormal['mse']), " std:", np.std(abnormal['mse']))

    tp = len(abnormal.loc[test_score_df['anomaly'] == True])
    fn = len(abnormal.loc[test_score_df['anomaly'] == False])
    tn = len(normal.loc[test_score_df['anomaly'] == False])
    fp = len(normal.loc[test_score_df['anomaly'] == True])

    print("TP ", tp, " TN ", tn, " FP ", fp, " FN", fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * ((pre * recall) / (pre + recall))
    print("Accuracy", acc, "Precision", pre, "Recall", recall, "F1-Score", fscore)

    #tn = len(normal.loc[test_score_df['anomaly2'] == False])
    #fp = len(normal.loc[test_score_df['anomaly2'] == True])
    #tp = len(abnormal.loc[test_score_df['anomaly2'] == True])
    #fn = len(abnormal.loc[test_score_df['anomaly2'] == False])

    #print("TP ", tp, " TN ", tn, " FP ", fp, " FN", fn)
    #acc = (tp + tn) / (tp + tn + fp + fn)
    #pre = tp / (tp + fp)
    #recall = tp / (tp + fn)
    #fscore = 2 * ((pre * recall) / (pre + recall))
    #print("Accuracy", acc, "Precision", pre, "Recall", recall, "F1-Score", fscore)
    return threshold


def timing_on_records(model, records):
    start = time.time()
    df = pd.read_csv(records)
    sum = 0
    count = 0
    normal, abnormal = 0, 0
    for row in range(df.shape[0]):
        X = df.iloc[row, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
        X = X.to_numpy().reshape(-1, 360, 1).astype('float32')
        #b, a = scipy.signal.butter(3, [0.03, 0.13], "band")
        #for x in range(X.shape[0]):
        #    signals = scipy.signal.filtfilt(b, a, X[x].flatten())
        #    X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        X_test_pred = model.predict(X, verbose=0)
        if np.abs(np.subtract(X, X_test_pred)).mean(axis=1) > threshold:
            abnormal += 1
        else:
            normal += 1
        current = time.time()
        #print(current-start)
        sum += current-start
        count += 1
        start = time.time()
    #annotations = df.iloc[:, 1]
    #y = np.array(annotations).reshape((-1, 1))
    print("Average run time for each record:", sum/count)
    print("Normal:", normal, "Abnormal:", abnormal)
    #X_test_mae = np.abs(np.subtract(X, X_test_pred)).mean(axis=1)
    #X_test_mse = np.square(np.subtract(X, X_test_pred)).mean(axis=1)

    #test_score_df = pd.DataFrame(index=range(len(X_test_mae)))
    #test_score_df['loss'] = X_test_mae
    #test_score_df['mse'] = X_test_mse
    #test_score_df['Annotation'] = y

    #annotations = ['N']
    #normal = test_score_df.loc[test_score_df['Annotation'].isin(annotations)]
    #abnormal = test_score_df.loc[~test_score_df['Annotation'].isin(annotations)]
    #threshold = np.mean(normal['loss']) + np.std(normal['loss'])

    #test_score_df['threshold'] = threshold
    #test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

    #threshold2 = np.mean(normal['mse']) + np.std(normal['mse'])
    #test_score_df['threshold2'] = threshold2
    #test_score_df['anomaly2'] = test_score_df.mse > test_score_df.threshold

    #tn = len(normal.loc[test_score_df['anomaly'] == False])
    #fp = len(normal.loc[test_score_df['anomaly'] == True])
    #tp = len(abnormal.loc[test_score_df['anomaly'] == True])
    #fn = len(abnormal.loc[test_score_df['anomaly'] == False])

    #print("TP ", tp, " TN ", tn, " FP ", fp, " FN", fn)
    #acc = (tp + tn) / (tp + tn + fp + fn)
    #pre = tp / (tp + fp)
    #recall = tp / (tp + fn)
    #fscore = 2 * ((pre * recall) / (pre + recall))
    #print("Accuracy", acc, "Precision", pre, "Recall", recall, "F1-Score", fscore)

def main():
    local_dir = os.path.abspath('')
    print(local_dir)

    autoencoder, encoder, decoder = create_model()
    train_on_records(local_dir, "training_by_classes.csv", "validating_by_classes.csv", autoencoder)
    #autoencoder.save("autoencoder.h5")
    #encoder.save("encoder.h5")
    #decoder.save("decoder.h5")
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()
    #with open('GRU_90_45.tflite', 'wb') as f:
    #    f.write(tflite_model)
    
    # print("Testing on: ", testing_set)
    #evaluate_on_records(model, "testing_by_classes.csv", local_dir)
    #interpreter = tf.lite.Interpreter(model_content=tflite_model)
    #lite_predict_on_record(interpreter, "testing_by_classes.csv", local_dir)
    #autoencoder = tf.keras.models.load_model("autoencoder.h5")
    #encoder = tf.keras.models.load_model("encoder.h5")
    #encoder.summary()
    threshold = evaluate_on_records(autoencoder, 'validating_by_classes.csv')
    predict_on_records(autoencoder, "testing_by_classes.csv", threshold)
    #timing_on_records(autoencoder, "testing_by_classes.csv", threshold)

if __name__ == '__main__':
    main()
