import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import normalize
import time
import tempfile
import pywt

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
    #model.add(keras.layers.Dense(45, activation='relu'))
    #model.add(keras.layers.LSTM(23))
    model.add(keras.layers.LSTM(360, return_sequences=True))
    model.add(keras.layers.LSTM(180))
    model.add(keras.layers.Dense(90, activation='relu'))
    model.add(keras.layers.Dense(len(CLASSIFICATION.keys()), activation=tf.nn.softmax, name='output'))

    
    dot_img_file = 'model_summary.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    lf = tf.keras.losses.CategoricalCrossentropy()
    #lf = tfa.losses.SigmoidFocalCrossEntropy(alpha=1.0, gamma=2.0)
    model.compile(optimizer='nadam',
                  loss=lf,
                  metrics=METRICS)

    model.summary()

    return model


def train_on_records(local_dir, training, validating, model):
    X, y = get_data_from_csv(training)
    vx, vy = get_data_from_csv(validating)
    history = model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE, validation_data=(vx, vy), shuffle=True)
    
    #print("Loss: ", history.history['loss'])
    #print("Val Loss: ", history.history['val_loss'])
    #print("Accuracy: ", history.history['accuracy'])
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title("Losses")
    plt.legend()
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.title("Accuracy")
    plt.show()


def pruning(training, validating, model):
    X, y = get_data_from_csv(training)
    vx, vy = get_data_from_csv(validating)

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 32
    epochs = 2

    num_signals = X.shape[0]
    end_step = np.ceil(num_signals / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                                   final_sparsity=0.80,
                                                                   begin_step=0,
                                                                   end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(X, y,
                      batch_size=batch_size, epochs=epochs, verbose=VERBOSE, validation_data=(vx, vy), shuffle=True,
                      callbacks=callbacks)
    return model_for_pruning


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
    y = []
    yp = []
    df = pd.read_csv(records)
    #df = df.iloc[:10, :]
    X = df.iloc[:, 360-int(WINDOW_SIZE/2):360+int(WINDOW_SIZE/2)]
    annotations = df.iloc[:, 1]
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
        mapping = map(mymap, annotations[x])
        anno =  np.array(list(mapping)).reshape(-1, 1)
        temp = keras.utils.to_categorical(anno, num_classes=len(CLASSIFICATION))
        y.append(temp)
        yp.append(model.predict(X[x].reshape((1, WINDOW_SIZE, 1))))
        current = time.time()
        #print(current-start)
        sum += current-start
        count += 1
        start = time.time()
    y = np.array(y).reshape(-1, len(CLASSIFICATION))
    yp = np.array(yp).reshape(-1, len(CLASSIFICATION))
    print("Average run time for each record:", sum/count)

    y_true = np.argmax(y, axis=1)
    mapping = map(reverse_map, y_true)
    y_true = np.array(list(mapping)).reshape(-1, 1)
    y_pred = np.argmax(yp, axis=1)
    mapping = map(reverse_map, y_pred)
    y_pred = np.array(list(mapping)).reshape(-1, 1)

    # print(y_true)
    # print(y_pred)
    #print(list(CLASSIFICATION.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSIFICATION.keys()))
    # print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASSIFICATION.keys())).plot()
    ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm), display_labels=list(CLASSIFICATION.keys())).plot()
    plt.show()
    print(classification_report(y_true, y_pred, zero_division=0))


def predict_on__nonannotated_record(model, records):
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
    y_pred = np.argmax(yp, axis=1)
    mapping = map(reverse_map, y_pred)
    y_pred = np.array(list(mapping)).reshape(-1, 1)
    print(y_pred)

    print("Average run time for each record:", sum/count)


def lite_predict_on_record(interpreter, records, local_dir):
    sum = 0
    count = 0
    start = time.time()
    y = np.array([[]])
    yp = np.array([[]])
    df = pd.read_csv(records)
    #df = df.iloc[:10, :]
    X = df.iloc[:, 360-int(WINDOW_SIZE/2):360+int(WINDOW_SIZE/2)]
    annotations = df.iloc[:, 1]
    X = X.to_numpy().reshape(-1, WINDOW_SIZE, 1)
    nyquist = 360/2
    high = 40/nyquist
    low = 0.5/nyquist
    b, a = scipy.signal.butter(5, [low, high], "band")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    for x in range(X.shape[0]):
        signals_butter = scipy.signal.filtfilt(b, a, X[x].flatten())
        signals = signals_butter.reshape(WINDOW_SIZE, 1)
        signals = normalize(signals, axis=0)
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        mapping = map(mymap, annotations[x])
        anno =  np.array(list(mapping)).reshape(-1, 1)
        temp = keras.utils.to_categorical(anno, num_classes=len(CLASSIFICATION))
        if y.shape[1] == 0:
            y = temp
        else:
            y = np.append(y, temp, axis=0)
        interpreter.set_tensor(input_details[0]['index'], X[x].reshape((-1, WINDOW_SIZE, 1)).astype('float32'))
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


def lite_predict_on__nonannotated_record(interpreter, records):
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
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    for x in range(X.shape[0]):
        signals_butter = scipy.signal.filtfilt(b, a, X[x].flatten())
        signals = signals_butter.reshape(WINDOW_SIZE, 1)
        signals = normalize(signals, axis=0)
        X[x] = signals.reshape(-1, WINDOW_SIZE, 1)
        interpreter.set_tensor(input_details[0]['index'], X[x].reshape((-1, WINDOW_SIZE, 1)).astype('float32'))
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
    local_dir = os.path.abspath('')

    #model = create_model()
    #train_on_records(local_dir, "training_by_classes.csv", "validating_by_classes.csv", model)
    #model.save("LSTM_360_180_90_l_then_d.h5")
    model = tf.keras.models.load_model("LSTM_90_45_norm.h5")
    #model = pruning("training_by_classes.csv", "validating_by_classes.csv", model)
    
    #predict_on_record(model, "testing_by_classes.csv", local_dir)
    predict_on__nonannotated_record(model, 'collab_360hz_data.csv')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    #with open('LSTM_360_180_90_l_then_d.tflite', 'wb') as f:
    #    f.write(tflite_model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    #lite_predict_on_record(interpreter, "testing_by_classes.csv", local_dir)
    lite_predict_on__nonannotated_record(interpreter, 'collab_360hz_data.csv')
    


if __name__ == '__main__':
    main()
