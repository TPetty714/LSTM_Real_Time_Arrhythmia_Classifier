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


def lite_predict_on_record(interpreter, records):
    sum = 0
    count = 0
    start = time.time()
    y = np.array([[]])
    yp = np.array([[]])
    df = pd.read_csv(records)
    #df = df.iloc[:10, :]
    X = df.iloc[:, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
    annotations = df.iloc[:, 1]
    X = X.to_numpy().reshape(-1, 360, 1)
    b, a = scipy.signal.butter(3, [0.03, 0.13], "band")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    for x in range(X.shape[0]):
        signals = scipy.signal.filtfilt(b, a, X[x].flatten())
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        mapping = map(mymap, annotations[x])
        anno =  np.array(list(mapping)).reshape(-1, 1)
        temp = keras.utils.to_categorical(anno, num_classes=len(CLASSIFICATION))
        if y.shape[1] == 0:
            y = temp
        else:
            y = np.append(y, temp, axis=0)
        interpreter.set_tensor(input_details[0]['index'], X[x].reshape((-1, 360, 1)).astype('float32'))
        interpreter.invoke()
        if yp.shape[1] == 0:
            #yp = model.predict(X[x].reshape((1, WINDOW_SIZE, 1)))
            yp = interpreter.get_tensor(output_details[0]['index'])
        else:
            yp = np.append(yp, interpreter.get_tensor(output_details[0]['index']), axis=0)
            #yp = np.append(yp, model.predict(X[x].reshape((1, WINDOW_SIZE, 1))), axis=0)
        current = time.time()
        #print(current-start)
        sum += current-start
        count += 1
        start = time.time()

    print("Average run time for each record:", sum/count)

    y_true = np.argmax(y, axis=1)
    mapping = map(reverse_map, y_true)
    y_true = np.array(list(mapping)).reshape(-1, 1)
    y_pred = np.argmax(yp, axis=1)
    mapping = map(reverse_map, y_pred)
    y_pred = np.array(list(mapping)).reshape(-1, 1)

    #cm = confusion_matrix(y_true, y_pred, labels=list(CLASSIFICATION.keys()))
    #ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASSIFICATION.keys())).plot()
    #ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=list(CLASSIFICATION.keys())).plot()
    #plt.show()

    print(classification_report(y_true, y_pred, zero_division=0))

    


def main():
    interpreter = tf.lite.Interpreter(model_path='LSTM_90_45.tflite')
    lite_predict_on_record(interpreter, "testing_by_classes.csv")


if __name__ == '__main__':
    main()
