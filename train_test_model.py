# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:22:21 2019

@author: awjde
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.mixture
import Preprocessing as pre
from sklearn import naive_bayes
import numpy as np
from imblearn.over_sampling import ADASYN
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import (Conv1D, MaxPool1D, Dense, Dropout,
                          Flatten, BatchNormalization, Activation)
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
from keras import backend as K
from sklearn.model_selection import train_test_split

def precision(y_true, y_pred):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    precision = tp / (tp + fp)
    return precision

def recall(y_true, y_pred):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    recall = tp / (tp + fn)
    return recall

def testOnHoldout(model, X_train, y_train, X_test, y_test):
    '''
    Use this function to test any sklearn model on the test set

    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    '''
    X_train, y_train = pre.oversample(X_train, y_train) #oversample
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy: ' + str(model.score(X_test, y_test)))
    print('Precision: ' + str(sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)))
    print('Recall: ' + str(sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)))
    print('F1: ' + str(sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred)))
    print('Confusion: ' + str(sklearn.metrics.confusion_matrix(y_test, y_pred)))
    print('')

def testModel(model, X, y):
    '''
    Perform cross validation on any sklearn model

    :param model:
    :param X:
    :param y:
    :return:
    None
    '''
    precision = []
    recall = []
    accuracies = []
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    i = 1
    for train_index, validation_index in skf.split(X, y):
        print('Fold ' + str(i) + "/10")
        i+=1
        #divide train and validation set
        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        X_train, y_train = pre.oversample(X_train, y_train) # oversample

        model.fit(X_train, y_train)

        #get metrics
        accuracy = model.score(X_validation, y_validation) # get accuracy
        y_pred = model.predict(X_validation)
        precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred)) #get precision
        recall.append(sklearn.metrics.recall_score(y_validation, y_pred)) #get recall
        accuracies.append(accuracy)

    #average metrics from each iteration
    avg_recall = np.average(recall)
    avg_precision = np.average(precision)
    accuracy = np.average(accuracies)
    F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) #compute f1 score
    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(avg_precision))
    print('Recall: ' + str(avg_recall))
    print('F1: ' + str(F1))
    print('')

def CNN(X_train, y_train, X_test, y_test):

    #oversample
    X_train, y_train = pre.oversample(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42,
                                                      stratify=y_train)

    #reshape the data to 3 dimensions so keras will accept it
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)


    #define the CNN
    cnn = Sequential()
    cnn.add(Conv1D(filters=8, kernel_size=10, activation='elu', input_shape=(X_train.shape[1], 1)))
    cnn.add(MaxPool1D(strides=6))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(filters=16, kernel_size=10, activation='elu'))
    cnn.add(MaxPool1D(strides=6))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(filters=32, kernel_size=10, activation='elu'))
    cnn.add(MaxPool1D(strides=6))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(filters=64, kernel_size=10, activation='elu'))
    cnn.add(MaxPool1D(strides=6))
    cnn.add(Flatten())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(1, activation='sigmoid'))


    cnn.compile(optimizer=Adam(4e-5), loss='binary_crossentropy', metrics=['accuracy']) #compile model

    es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=50) #setup early stopping
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True) #saves best model

    #fit model
    history = cnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, mc], verbose=0)

    saved_cnn = load_model('best_model.h5')

    y_pred = saved_cnn.predict_classes(X_test)

    #print metrics
    print('Accuracy: ' + str(sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Precision: ' + str(sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)))
    print('Recall: ' + str(sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)))
    print('F1: ' + str(sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred)))
    print('Confusion: ' + str(sklearn.metrics.confusion_matrix(y_test, y_pred)))
    print('')

def runSVM():
    '''
    Runs the SVm grid search
    '''
    X, y = pre.loadData('norm_detrend_train')
    kernel = ['rbf', 'sigmoid', 'linear'] #different kernels to try
    C = [1, 10, 100] # different values of C to try
    gamma = ['auto']
    output = []
    z = 0
    for k in kernel:
        for g in gamma:
            for c in C:
                print(z)
                z+=1
                out = []
                precision = []
                recall = []
                accuracies = []
                skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
                for train_index, validation_index in skf.split(X, y):
                    model = sklearn.svm.SVC(kernel=k, gamma=g, C=c)

                    # divide train and validation set
                    X_train, X_validation = X[train_index], X[validation_index]
                    y_train, y_validation = y[train_index], y[validation_index]

                    X_train, y_train = pre.oversample(X_train, y_train)

                    model.fit(X_train, y_train)

                    #get metrics
                    accuracy = model.score(X_validation, y_validation)
                    y_pred = model.predict(X_validation)
                    precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred))
                    recall.append(sklearn.metrics.recall_score(y_validation, y_pred))
                    accuracies.append(accuracy)


                #average all the metrics from each iteration
                avg_recall = np.average(recall)
                avg_precision = np.average(precision)
                accuracy = np.average(accuracies)
                F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                out.append(avg_recall)
                out.append(np.std(recall))
                out.append(avg_precision)
                out.append(np.std(precision))
                out.append(F1)
                out.append(accuracy)
                out.append(np.std(accuracies))
                output.append(out)
    np.savetxt("Results/SVM_GridSearchNorm.csv", output, delimiter=',') #save grid search results

def runRandomForest():
    '''
    perform random forest grid search
    '''
    X, y = pre.loadData('norm_detrend_train')

    #set the different parameters to try
    max_depth = [10, 30, 50, 70, 110]
    min_samples_split = [1, 2, 5, 10]
    n_estimators = [100, 200, 300, 400, 500, 600, 700, 800]


    output = []
    z = 0
    for d in max_depth:
        for s in min_samples_split:
            for n in n_estimators:
                print("Random Forest" + str(z))
                z+=1
                out = []
                precision = []
                recall = []
                accuracies = []
                skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
                for train_index, validation_index in skf.split(X, y):
                    model = sklearn.ensemble.RandomForestClassifier(max_depth=d, min_samples_split=s, n_estimators=n)

                    # divide train and validation set
                    X_train, X_validation = X[train_index], X[validation_index]
                    y_train, y_validation = y[train_index], y[validation_index]

                    X_train, y_train = pre.oversample(X_train, y_train)

                    model.fit(X_train, y_train)

                    #get metrics
                    accuracy = model.score(X_validation, y_validation)
                    y_pred = model.predict(X_validation)
                    precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred))
                    recall.append(sklearn.metrics.recall_score(y_validation, y_pred))
                    accuracies.append(accuracy)

                #average all the metrics from each iteration
                avg_recall = np.average(recall)
                avg_precision = np.average(precision)
                accuracy = np.average(accuracies)
                F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                out.append(avg_recall)
                out.append(np.std(recall))
                out.append(avg_precision)
                out.append(np.std(precision))
                out.append(F1)
                out.append(accuracy)
                out.append(np.std(accuracies))
                output.append(out)
    np.savetxt("Results/RandomForest_GridSearch.csv", output, delimiter=',') #save grid search results

def runDecisionTree():
    '''
    perform grid search on decision tree
    '''
    X, y = pre.loadData('norm_detrend_train')

    #set the different parameters to try
    max_depth = [10, 30, 50, 70, 110]
    min_samples_split = [1, 2, 5, 10]
    output = []
    z = 0
    for d in max_depth:
        for s in min_samples_split:
            z+=1
            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
            precision = []
            recall = []
            accuracies = []
            out = []
            for train_index, validation_index in skf.split(X, y):
                model = sklearn.tree.DecisionTreeClassifier(max_depth=d, min_samples_split=s)

                # divide train and validation set
                X_train, X_validation = X[train_index], X[validation_index]
                y_train, y_validation = y[train_index], y[validation_index]

                X_train, y_train = pre.oversample(X_train, y_train)

                model.fit(X_train, y_train)

                # get metrics
                accuracy = model.score(X_validation, y_validation)
                y_pred = model.predict(X_validation)
                precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred))
                recall.append(sklearn.metrics.recall_score(y_validation, y_pred))
                accuracies.append(accuracy)

            # average all the metrics from each iteration
            avg_recall = np.average(recall)
            avg_precision = np.average(precision)
            accuracy = np.average(accuracies)
            F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            out.append(avg_recall)
            out.append(np.std(recall))
            out.append(avg_precision)
            out.append(np.std(precision))
            out.append(F1)
            out.append(accuracy)
            out.append(np.std(accuracies))
            output.append(out)
    np.savetxt("Results/DecisionTree_GridSearch.csv", output, delimiter=',') #save grid search results

def runNiaveBayes():
    '''
    Perform cross validation on naive bayes
    '''
    output = []
    X, y = pre.loadData('norm_detrend_train')

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    precision = []
    recall = []
    accuracies = []
    out = []
    for train_index, validation_index in skf.split(X, y):
        model = naive_bayes.GaussianNB()

        #divide train and validation set
        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        X_train, y_train = pre.oversample(X_train, y_train)

        model.fit(X_train, y_train)

        #get metrics
        accuracy = model.score(X_validation, y_validation)
        y_pred = model.predict(X_validation)
        precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred))
        recall.append(sklearn.metrics.recall_score(y_validation, y_pred))
        accuracies.append(accuracy)

    # average all the metrics from each iteration
    avg_recall = np.average(recall)
    avg_precision = np.average(precision)
    accuracy = np.average(accuracies)
    F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    out.append(avg_recall)
    out.append(np.std(recall))
    out.append(avg_precision)
    out.append(np.std(precision))
    out.append(F1)
    out.append(accuracy)
    out.append(np.std(accuracies))
    output.append(out)
    np.savetxt("Results/NiaveBayes_GridSearch.csv", output, delimiter=',') #save grid search results

def runKNN():
    output = []
    X, y = pre.loadData('norm_detrend_train')

    #set the different parameters to try
    n_neighbors = [i for i in range(5, 120, 5)]
    P = [1, 2]
    z = 0
    for n in n_neighbors:
        for p in P:
            print("KNN " + str(z))
            z += 1
            out = []
            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
            precision = []
            recall = []
            accuracies = []
            for train_index, validation_index in skf.split(X, y):
                model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, algorithm='brute', p=p)

                # divide train and validation set
                X_train, X_validation = X[train_index], X[validation_index]
                y_train, y_validation = y[train_index], y[validation_index]

                X_train, y_train = pre.oversample(X_train, y_train)

                model.fit(X_train, y_train)

                #get metrics
                accuracy = model.score(X_validation, y_validation)
                y_pred = model.predict(X_validation)
                precision.append(sklearn.metrics.average_precision_score(y_validation, y_pred, pos_label=1))
                recall.append(sklearn.metrics.recall_score(y_validation, y_pred, pos_label=1))
                accuracies.append(accuracy)

            # average all the metrics from each iteration
            avg_recall = np.average(recall)
            avg_precision = np.average(precision)
            accuracy = np.average(accuracies)
            F1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            out.append(avg_recall)
            out.append(np.std(recall))
            out.append(avg_precision)
            out.append(np.std(precision))
            out.append(F1)
            out.append(accuracy)
            out.append(np.std(accuracies))
            output.append(out)
    np.savetxt("Results/KNN_GridSearch.csv", output, delimiter=',') #save gridsearch results




