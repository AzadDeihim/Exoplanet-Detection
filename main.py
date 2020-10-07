import Preprocessing
from train_test_model import testModel
import train_test_model
from train_test_model import testOnHoldout
import sklearn
import matplotlib.pyplot as plt
from Visualize import runTSNE
import numpy as np
from sklearn.model_selection import train_test_split
import time
import train_test_model


#preprocess and format data
Preprocessing.csvToNpy() #converts the data from csv format to numpy arrays and saves then as raw_train and raw_test

X_train_raw, y_train_raw = Preprocessing.loadData('raw_train') #load the train data
X_test_raw, y_test_raw = Preprocessing.loadData('raw_test') #load the test data

X_train_new = Preprocessing.reduce_upper_outliers(Preprocessing.detrender_normalizer(X_train_raw)) #detrend, normalize, and remove upper outliers from train data
Preprocessing.saveData(X_train_new, y_train_raw, 'norm_detrend_train') #save the preprocessed train data

X_test_new = Preprocessing.reduce_upper_outliers(Preprocessing.detrender_normalizer(X_test_raw)) #detrend, normalize, and remove upper outliers from test data
Preprocessing.saveData(X_test_new, y_test_raw, 'norm_detrend_test') #save the preprocessed test data

#uncomment if you want to generate the tSNE plots but it will take over an hour
#runTSNE()

#load preprocessed train and test sets
X_train, y_train = Preprocessing.loadData('norm_detrend_train')
X_test, y_test = Preprocessing.loadData('norm_detrend_test')


#produce a plot of exoplanet 11 before and after preprocessing
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(X_train_raw[11])
ax[0].title.set_text('Unprocessed Signal (Exoplanet 11)')
ax[0].set_ylabel('Flux')
ax[0].set_xlabel('Time Step')
plt.ylabel('Flux')
plt.xlabel('Time Step')
ax[1].plot(X_train[11])
ax[1].title.set_text('Preprocessed Signal (Exoplanet 11)')
ax[1].set_ylabel('Flux')
ax[1].set_xlabel('Time Step')
plt.tight_layout()
plt.show()

#Redefine the train and test set using an 80/20 split rather than 90/10
X = np.vstack((X_train, X_test))
y = np.append(y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
Uncomment if you would like to run the grid searches, but note it will take about 3-5 days to complete

train_test_model.runNiaveBayes()
train_test_model.runKNN()
train_test_model.runDecisionTree()
train_test_model.runRandomForest()

'''

#perform cross-val on the baseline
print('Baseline')
model = sklearn.linear_model.LogisticRegression(solver='liblinear')
print("Cross-Validation")
testModel(model, X_train_raw, y_train_raw)



#perform cross-val and test on the test set using KNN
print('K-Nearest Neighbors')
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='brute', p=1)
print("Cross-Validation")
testModel(knn, X_train, y_train)
print("Testing on holdout")
testOnHoldout(knn, X_train, y_train, X_test, y_test)


#perform cross-val and test on the test set using Random Forest
print('Random Forest')
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=400, min_samples_split=5, max_depth=50)
print("Cross-Validation")
testModel(rf, X_train, y_train)
print("Testing on holdout")
testOnHoldout(rf, X_train, y_train, X_test, y_test)

#perform cross-val and test on the test set using Decison Tree
print('Decision Tree')
dt = sklearn.tree.DecisionTreeClassifier(max_depth=70, min_samples_split=5)
print("Cross-Validation")
testModel(dt, X_train, y_train)
print("Testing on holdout")
testOnHoldout(dt, X_train, y_train, X_test, y_test)


#perform cross-val and test on the test set using Naive Bayes
print('Naive Bayes')
nb = sklearn.naive_bayes.GaussianNB()
print("Cross-Validation")
testModel(nb, X_train, y_train)
print("Testing on holdout")
testOnHoldout(nb, X_train, y_train, X_test, y_test)


#perform cross-val and test on the test set using CNN
print('Convolutional Neural Network')
print("Testing on holdout")
train_test_model.CNN(X_train, y_train, X_test, y_test)

