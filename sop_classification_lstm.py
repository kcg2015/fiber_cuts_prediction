# -*- coding: utf-8 -*-


import numpy as np
import keras
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from time import time


def create_model(neurons, input_shape =[3, 80], drop_rate = 0.1, 
                 recu_drop_rate = 0.01):
    
    '''
    Function that creates a LSTM model 
    Input:
        neurons:        layer size
        input_shape:    dimension of the input vector(matrix)
        drop_rate:      dropout reate
        recu_drop_rate: recurrent dropout rate
    Output:
        model: created LSTM model per the specificed model parameters
    '''
    
    
    keras.backend.clear_session()    
    model = Sequential()
    model.add(LSTM(neurons, input_shape = input_shape, dropout= drop_rate, 
                   recurrent_dropout = recu_drop_rate, stateful=False))  
    model.add(Dense(4, activation = 'softmax'))    
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def lstm_grid_search(X_train, y_train, tst_size, r_state, neurons, drop_rate, 
                      recu_drop_rate, epochs, batch_size):
    
    '''
    Function that carries out grid search to find the best hyperparameters
    Input:
        X_train, y_train: data and label
        tst_size:    % of data used for validation evaluation during grid search
        r_state:  random state for validation split
        neurons:        layer size
        input_shape:    dimension of the input vector(matrix)
        drop_rate:      dropout reate
        recu_drop_rate: recurrent dropout rate
        batch_size:  batch size used for training
        
    Output:
        return grid search results
        
    '''
    
    cv = StratifiedShuffleSplit(n_splits = 5, test_size = tst_size, random_state = r_state)
    param_grid = dict(neurons = neurons, drop_rate = drop_rate, recu_drop_rate = recu_drop_rate, 
                      epochs = epochs, batch_size = batch_size)
    model = KerasClassifier(build_fn = create_model, epochs = epochs, batch_size = batch_size, verbose=1)
    grid = GridSearchCV(estimator = model, param_grid = param_grid,  cv = cv, verbose = 2)
    grid.fit(X_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    return [grid.best_params_, grid.best_score_]
    


def lstm_train_test(X_train, y_train, X_test, best_para):
    
    '''
    Function that carries out grid search to find the best hyperparameters
    Input:
        X_train, y_train: training data and label
        X_test:  test data 
        best_para: parameters obtained from grid search
        
    Output:
        return y_hat: prediction
        
    '''
    
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    neurons = best_para['neurons']
    drop_rate = best_para['drop_rate']
    recu_drop_rate = best_para['recu_drop_rate']
    epochs = best_para['epochs']
    batch_size = best_para['batch_size']
    model = create_model(neurons = neurons, drop_rate = drop_rate, 
                         recu_drop_rate = recu_drop_rate)
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
    
    y_hat = model.predict(X_test)
    
    return y_hat

def run(tst_sizes, best_para, source):
    
    '''
    Input:
        tst_sizes: fraction of data used for testing
        best_para: parameters as the result of grid search
        source: flag for the file names to save the results
    
    Output: average [accuracy, precision, recall]
    '''
    
    
    tst_sz1, tst_sz2 = tst_sizes[0], tst_sizes[1]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_local_dir = '/datasets/'
    
    if source == 1:
       npz_file = 'mvt_dsf200_rot.npz'
       source_str = '_rot_'
    elif source == 2:   
        npz_file = 'mvt_dsf200_no_rot.npz'
        source_str = '_no_rot_'
    data = np.load(dir_path + dataset_local_dir + npz_file)
    np_array = data['X']
    
    np_array = np_array[~np.isnan(np_array).any(axis=1)]
    
    X = np_array[:, : -1]
    Y = np_array[:, -1]
    
    no_samples, vec_len = X.shape
    X = X.reshape([no_samples, 3, 80])
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    Y = dummy_y
    
   

    Y_test_hat = [ ]
    metrics =[]
    for seed in [22, 66, 345, 536, 9999, 888, 1, 0, 12345, 23]:
        # Loop through all the seeds for calulating the average of the accuracy
        X, Y = shuffle(X, Y, random_state = seed + 2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = tst_sz1, random_state = seed)
        X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size = tst_sz2, random_state = seed + 4)
        Y_hat = lstm_train_test(X_train, Y_train, X_test, best_para)
        Y_hat_label = np.argmax(Y_hat, axis=1)
        Y_test_label = np.argmax(Y_test, axis=1)
        Y_test_hat.append(Y_test_label)
        Y_test_hat.append(Y_hat_label)
        accu = accuracy_score(Y_test_label, Y_hat_label)
        precision, recall, f1, support = precision_recall_fscore_support(Y_test_label, Y_hat_label, average='weighted')
        metrics.append([accu, precision, recall, f1])         
    
    Y_test_hat_np = np.array(Y_test_hat)
    metrics_np = np.array(metrics)
    metrics_avg = np.mean(metrics_np, axis = 0)
    
    Y_file_name = 'Y_lstm' + source_str + str(X_train.shape[0])+'.npz'
    metric_file_name = 'metric_lstm'+ source_str + str(X_train.shape[0])+'.npz'
    
    np.savez(Y_file_name, X = Y_test_hat_np)
    np.savez(metric_file_name, X = metrics_np) 
    
    return metrics_avg

if __name__ == '__main__':
    
    
    start = time()
    
    avg_list = []
    
    
    best_para = {'neurons': 512, 'drop_rate': 0.2, 
                 'recu_drop_rate':0.05, 'batch_size': 64, 'epochs': 20}
    for size in [[0.2, 0.8]]:   
         
         avg = run(size, best_para, 1)
         avg_list.append(avg)
    
    end = time()
    
    print(avg_list)     
    print((end - start)/3600.0)
         

