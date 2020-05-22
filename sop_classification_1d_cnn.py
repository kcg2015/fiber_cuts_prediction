import os
from time import time
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import np_utils
from tensorflow import set_random_seed

LABELS = ['mvt1', 'mvt2', 'mvt3', 'mvt4']

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_axis(ax, x, y, title):

    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def create_1D_CNN_model(para_model):
    
    '''
    Function that creates a 1-D CNN model 
    Input:
        filters:        
        input_shape:    dimension of the input vector(matrix)
        drop_rate:      dropout reate
    Output:     
        model: created 1-D CNN model per the specificed model parameters
    '''
    FILTERS = para_model['filters']
    INPUT_SHAPE = para_model['input_shape']
    TIME_STEPS = para_model['time_steps']
    DIM = para_model['dim']
    DROP_RATE = para_model['drop_rate']
    NUM_CLASSES = para_model['num_classes']
    keras.backend.clear_session()
    model = Sequential()
    model.add(Reshape((TIME_STEPS, DIM), input_shape = INPUT_SHAPE))
    model.add(Conv1D(FILTERS[0][0], FILTERS[0][1], activation='relu', input_shape=(TIME_STEPS, DIM)))
    model.add(Conv1D(FILTERS[1][0], FILTERS[1][1], activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(FILTERS[2][0], FILTERS[2][1], activation='relu'))
    model.add(Conv1D(FILTERS[3][0], FILTERS[3][1], activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(DROP_RATE))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])
    
    return model

def plot_history(history):
    print("\n--- Learning curve of model training ---\n")
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

def train_1D_CNN_model(model, X_train, y_train, X_test, train_paras, plot_hist = True):
    
    '''
    Function that trains the 1-D CNN model 
    Input:
        model:          created 1-D CNN model
        X_train, y_train, X_test: training data, train, label
        train_paras: a dictionary that holds related training hyper parameters
        plot_hist: whether plot the training history or not, default is set to be True
    Output:     
        model: trained 1-D CNN model per the specificed training parameters
    '''
    
    callbacks_list = [
     keras.callbacks.ModelCheckpoint(
         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
         monitor='val_loss', save_best_only=False),
     keras.callbacks.EarlyStopping(monitor='acc', patience=1)]
 
    # Hyper-parameters
    BATCH_SIZE = train_paras['batch_size'] #20, 400
    EPOCHS = train_paras['epochs']  #50
    VALIDATION_SPLIT = train_paras['validation_split']
     
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model.fit(X_train,
                           y_train,
                           batch_size = BATCH_SIZE,
                           epochs = EPOCHS,
                           callbacks = None,
                           validation_split = VALIDATION_SPLIT,
                           verbose=1)
     
    if plot_hist:
        plot_history(history)
        
    return model, callbacks_list   


def run(run_paras):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_local_dir = run_paras['dataset_dir'] #'/Data/'
    npz_file = run_paras['data_file'] # mvt_dsf200_no_rot.npz'
    data = np.load(dir_path + dataset_local_dir + npz_file)
    np_array = data['X']
    np_array = np_array[~np.isnan(np_array).any(axis=1)]
    X = np_array[:, : -1]
    Y = np_array[:, -1]
    
    no_samples, vec_len = X.shape
    
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    Y = dummy_y
    seed = 23
    X, Y = shuffle(X, Y, random_state = seed + 2)
    print(X.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed)
    X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size = 0.8, random_state = seed + 4)
    
    print("\n--- Reshape data to be accepted by Keras ---\n")
      # Inspect x data
    print('x_train shape: ', X_train.shape)
    print(X_train.shape[0], 'training samples')
    # Inspect y data
    print('Y_train shape: ', Y_train.shape)
    print("\n--- Create neural network model ---\n")
    
    
    set_random_seed(2)
    model_paras = {
    'filters': [[100, 10], [100, 10], [160, 10], [160, 10]],
    'input_shape': (240, ),
    'time_steps': 80,
    'dim': 3,
    'drop_rate': 0.2,
    'num_classes': 4}
    
    model = create_1D_CNN_model(model_paras)
    print(model.summary())
    
    print("\n--- Fit the model ---\n")
    
    train_paras = {'batch_size': 32, 'epochs': 50, 'validation_split': 0.2}
     
    model, callbacks_list = train_1D_CNN_model(model, X_train, Y_train, X_test, train_paras, plot_hist = True)
    
    print("\n--- Check against test data ---\n")
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("\nAccuracy on test data: %0.3f" % score[1])
    print("\nLoss on test data: %0.3f" % score[0])
    print("\n--- Confusion matrix for test data ---\n")
    
    start = time()
    #Y_pred_test = model_m.predict(X_test[0, :].reshape(-1, 1).T)
     # Take the class with the highest probability from the test predictions
    Y_pred_test = model.predict(X_test) 
    max_y_pred_test = np.argmax(Y_pred_test, axis=1)
    max_y_test = np.argmax(Y_test, axis=1)
    end = time()
    show_confusion_matrix(max_y_test, max_y_pred_test)
    print(end - start)
    print("\n--- Classification report for test data ---\n")
    print(classification_report(max_y_test, max_y_pred_test))
     
    
    
if __name__ == '__main__':
    
    run_paras = {
           'dataset_dir': '/Data/',
           'data_file':  'mvt_dsf200_no_rot.npz' 
           }
    run(run_paras)    