import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time

# Customizable Values
WINDOW_SIZE = 360
EPOCHS = 50
VERBOSE = 2
CLASSIFICATION = {'N': 0, 'L': 1, 'R': 2, 'B': 3, 'A': 4, 'a': 5, 'J': 6, 'S': 7, 'V': 8,
                   'r': 9, 'F': 10, 'e': 11, 'j': 12, 'n': 13, 'E': 14, '/': 15, 'f': 16, 'Z': 17}
#CLASSIFICATION = {'N': 0, 'V': 1, 'F': 2, 'O': 3, 'E': 4, 'P': 5, 'Q': 6, 'Z': 7}
#CLASSIFICATION = {'N': 0, 'Z':1}

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
    X = df.iloc[:, WINDOW_SIZE-int(WINDOW_SIZE/2):WINDOW_SIZE+int(WINDOW_SIZE/2)]
    X = X.to_numpy().reshape(-1, 360, 1)
    b, a = scipy.signal.butter(3, [0.03, 0.13], "band")
    for x in range(X.shape[0]):
        signals = scipy.signal.filtfilt(b, a, X[x].flatten())
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
    annotations = df.iloc[:, 1]
    # print(annotations)
    mapping = map(mymap, annotations)
    annotations = np.array(list(mapping)).reshape(-1, 1)
    # print(annotations)
    y = keras.utils.to_categorical(annotations, num_classes=len(CLASSIFICATION))
    return X, y


def mymap(n):
    #if n in ['.', 'N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j']:
    #    n = 'N'
    #elif n in ['F', 'f']:
    #    n = 'F'
    #elif n in ['!', 'p']:
    #    n = 'O'
    #elif n in list(CLASSIFICATION.keys()):
    if n in list(CLASSIFICATION.keys()):
        n = n
    else:
        n = 'Z'
    return CLASSIFICATION[n]


def reverse_map(n):
    classes = list(CLASSIFICATION.keys())
    return classes[n]


def create_model():
    model = keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(WINDOW_SIZE, 1), name='input'))
    model.add(keras.layers.Dense(360, activation='relu'))
    model.add(keras.layers.LSTM(45))
    model.add(keras.layers.Dense(len(CLASSIFICATION.keys()), activation=tf.nn.softmax, name='output'))

    
    dot_img_file = 'model_summary.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    lf = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer='adam',
                  loss=lf,
                  metrics=METRICS)

    model.summary()

    return model


def train_on_records(local_dir, training, validating, model):
    X, y = get_data_from_csv(training)
    vx, vy = get_data_from_csv(validating)
    history = model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE, validation_data=(vx, vy))
    
    print("Loss: ", history.history['loss'])
    print("Val Loss: ", history.history['val_loss'])
    print("Accuracy: ", history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title("Losses")
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.title("Accuracy")
    plt.show()


def evaluate_on_records(model, records, local_dir):
    X, y = get_data_from_csv(records)
    evaluation_history = model.evaluate(X, y, verbose=VERBOSE)
    print("Evaluation", evaluation_history)
    tp = evaluation_history[1]
    fp = evaluation_history[2]
    tn = evaluation_history[3]
    fn = evaluation_history[4]
    return tp, fp, tn, fn


def predict_on_record(model, records, local_dir):
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
        if yp.shape[1] == 0:
            yp = model.predict(X[x].reshape((1, WINDOW_SIZE, 1)))
        else:
            yp = np.append(yp, model.predict(X[x].reshape((1, WINDOW_SIZE, 1))), axis=0)
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

    # print(y_true)
    # print(y_pred)
    print(list(CLASSIFICATION.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSIFICATION.keys()))
    # print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASSIFICATION.keys())).plot()
    ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=list(CLASSIFICATION.keys())).plot()
    plt.show()
    print(classification_report(y_true, y_pred, zero_division=0))


def lite_predict_on_record(interpreter, records, local_dir):
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

    # print(y_true)
    # print(y_pred)
    print(list(CLASSIFICATION.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSIFICATION.keys()))
    # print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASSIFICATION.keys())).plot()
    ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=list(CLASSIFICATION.keys())).plot()
    plt.show()
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    local_dir = os.path.abspath('')
    print(local_dir)
    # records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
    #            200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    # training_set, testing_set = Split(records)
    # training_set = [112, 113, 207, 217, 121, 105, 103, 101, 111, 116, 100, 221, 201, 215, 234, 114,
    #                 108, 212, 228, 209, 203, 231, 222, 219, 124, 118, 109, 107, 210, 208, 123, 122, 223, 230]
    # testing_set = [119, 213, 205, 233, 202, 200, 220, 115, 117, 232, 106, 214]
    # valitation_set = [116, 203, 114, 231, 105, 108, 209, 230, 201]
    # training_set, valitation_set = Split(training_set)
    # print("Training on", training_set)
    # print("Validating on", valitation_set)

    model = create_model()
    train_on_records(local_dir, "training_by_classes.csv", "validating_by_classes.csv", model)
    model.save("LSTM_360_45.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('LSTM_360_45.tflite', 'wb') as f:
        f.write(tflite_model)
    #model = tf.keras.models.load_model("LSTM_Targetted_Classification_"+str(WINDOW_SIZE)+"_e"+str(EPOCHS)+".h5")
    # print("Testing on: ", testing_set)
    #evaluate_on_records(model, "testing_by_classes.csv", local_dir)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    lite_predict_on_record(interpreter, "testing_by_classes.csv", local_dir)
    predict_on_record(model, "testing_by_classes.csv", local_dir)


if __name__ == '__main__':
    main()
