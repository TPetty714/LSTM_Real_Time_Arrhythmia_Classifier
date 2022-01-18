import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.metrics import classification_report
import time

# Customizable Values
WINDOW_SIZE = 360
EPOCHS = 30
VERBOSE = 0
CLASSIFICATION = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                   'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}


def get_data_from_csv(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
    X = X.to_numpy().reshape(-1, 360, 1)
    b, a = scipy.signal.butter(3, [0.03, 0.13], "band")
    for x in range(X.shape[0]):
        signals = scipy.signal.filtfilt(b, a, X[x].flatten())
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
    annotations = df.iloc[:, 1]
    mapping = map(mymap, annotations)
    annotations = np.array(list(mapping)).reshape(-1, 1)
    y = keras.utils.to_categorical(annotations, num_classes=len(CLASSIFICATION))
    return X, y


def mymap(n):
    if n in list(CLASSIFICATION.keys()):
        n = n
    else:
        n = 'Z'
    return CLASSIFICATION[n]


def reverse_map(n):
    classes = list(CLASSIFICATION.keys())
    return classes[n]


def lite_predict_on_nonannotated_record(interpreter, records):
    sum = 0
    count = 0
    start = time.time()
    yp = np.array([[]])
    df = pd.read_csv(records)
    #df = df.iloc[:10, :]
    X = df.iloc[:, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
    X = X.to_numpy().reshape(-1, 360, 1)
    nyquist = 360/2
    high = 40/nyquist
    low = 0.5/nyquist
    b, a = scipy.signal.butter(5, [low, high], "band")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    for x in range(X.shape[0]):
        signals = scipy.signal.filtfilt(b, a, X[x].flatten())
        signals = signals.reshape(WINDOW_SIZE, 1)
        signals = normalize(signals, axis=0)
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        interpreter.set_tensor(input_details[0]['index'], X[x].reshape((-1, 360, 1)).astype('float32'))
        interpreter.invoke()
        if yp.shape[1] == 0:
            yp = interpreter.get_tensor(output_details[0]['index'])
        else:
            yp = np.append(yp, interpreter.get_tensor(output_details[0]['index']), axis=0)
        current = time.time()
        #print(current-start)
        sum += current-start
        count += 1
        start = time.time()
    y_pred = np.argmax(yp, axis=1)
    mapping = map(reverse_map, y_pred)
    y_pred = np.array(list(mapping)).reshape(-1, 1)
    print(y_pred)

    print("Average run time for each record:", sum/count)

    


def main():
    interpreter = tf.lite.Interpreter(model_path='LSTM_90_45.tflite')
    lite_predict_on_nonannotated_record(interpreter, 'collab_500hz_data.csv')


if __name__ == '__main__':
    main()
