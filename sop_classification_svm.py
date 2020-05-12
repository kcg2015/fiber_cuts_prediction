# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score 


def SVM_grid_search(X, y, tst_size, r_state, C_range=np.logspace(-2, 2, 5),
                     gamma_range = np.logspace(-2, 2, 5), vb = 2):
    '''
    Function for SVM grid search
    
    X: Data
    y: label
    tst_size: fraction of data used for (validation) evaluation
    r_state: random seed for repetability
    C_range: search grid range for parameter C
    gamma_range: search grid range for parameter gamma
    
    '''
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=tst_size, random_state= r_state)
    grid = GridSearchCV(SVC(class_weight = 'balanced'), param_grid=param_grid, cv=cv, verbose = vb)
    grid.fit(X, y)
    
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    return [grid.best_params_, grid.best_score_]


def SVM_train_test(X_train, y_train, X_test, y_test, best_para):
    
    
    '''
    Function for SVM train and testing
    
    X_train: training data
    y_train: training label
    X_test: test data
    y_test: test label
    best_para: dictionary for best parameter (C and gamma)
    '''
    
    C_best, gamma_best = best_para['C'], best_para['gamma']
    
    model = SVC(C = C_best, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma = gamma_best, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    
    model.fit(X_train, y_train)
    
    y_hat = model.predict(X_test)

    return y_hat


def SVM_pipeline(X, y, test_sizes, r_states):
    
    '''
    SVM grid search, training and testing pipeline
    
    X: data
    y: label
    test_sizes: fractions for data split
                test_sizes[0]: fraction for first train/test split
                test_sizes[1]: fraction of data used for (validation) evaluation
    r_states:  random seeds used at various stages
    best_para: dictionary for best parameter (C and gamma)
    '''
    
    X, y = shuffle(X, y, random_state = r_states[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sizes[0], random_state = r_states[1])
    
    best_para, _ = SVM_grid_search(X_train, y_train, test_sizes[1], r_states[2], vb=2)
   
    y_hat = SVM_train_test(X_train, y_train, X_test, y_test, best_para)
    
    return y_hat, y_test



if __name__ == '__main__':

    data=np.load("/Users/kyleguan/Documents/SOP_classification_paper/python_script/datasets/mvt_dsf200_no_rot.npz")
    np_array = data['X']
    X = np_array[:, :-1]  
    y = np_array[:, -1]
    
    y_hat, y_test = SVM_pipeline(X, y, [0.6, 0.1], [88, 888, 8888])
    
    print("The accuracy: ", accuracy_score(y_test, y_hat) * 100)
    print("The classification report:")
    print(classification_report(y_test, y_hat))
    print("The confusion matrix: ")
    print(confusion_matrix(y_test, y_hat))
