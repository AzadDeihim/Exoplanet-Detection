import numpy as np
from scipy import stats as ss
from tsfresh.feature_extraction import feature_calculators as ts
from scipy import ndimage
from sklearn.model_selection import train_test_split

def csvToNpy():
    '''
    This function converts the original dataset from a .csv to a numpy array and
    saves them as 'raw_train' and 'raw_test'

    :return:
    None
    '''
    #load dataset
    train = np.genfromtxt('./kepler-labelled-time-series-data/exoTrain.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt('./kepler-labelled-time-series-data/exoTest.csv', delimiter=',', skip_header=1)

    #remove the first column, which contains labels
    X_train = train[:, 1:(len(train[0])-1)]
    X_test = test[:, 1:(len(test[0])-1)]

    #extract labels column, subtract by 1 to make the labels 0 and 1 rather than 1 and 2
    y_train = train[:, 0] - 1
    y_test = test[:, 0] - 1

    #save data as a .mat
    saveData(X_train, y_train, data_name='raw_train')
    saveData(X_test, y_test, data_name='raw_test')

def saveData(X, y, data_name='temp'):
    """
    Save data & labels as a .mat file in folder called preprocessedData
    """
    import scipy.io as io
    file_path = './Data/%s' % data_name

    dictionary = {}
    dictionary['data'] = X
    dictionary['labels'] = y

    io.savemat(file_path, dictionary)

def loadData(data_name='temp'):
    """
    returns
        (data, labels)
    """
    import scipy.io as sio
    file_path = './Data/%s' % data_name


    dictionary = sio.loadmat(file_name=file_path)
    X = dictionary['data']
    y = dictionary['labels']

    # saving arrays of shape (n,) in .mat converts them to shape (1, n)
    if y.shape[0] == 1:
        y = np.reshape(y,(y.shape[1],))

    return np.array(X), np.array(y)

def oversample(X, y):
    """
        https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html#imblearn.over_sampling.ADASYN
        """
    from imblearn.over_sampling import ADASYN
    aos = ADASYN()
    X_resampled, y_resampled = aos.fit_resample(X, y)

    return X_resampled, y_resampled

def detrender_normalizer(X):
    '''
    Detrend and normalize the signal

    :param X:

    :return:
    The detrended and normalized signal
    '''
    X2 = X - ndimage.filters.gaussian_filter(X, sigma=10)
    return (X2 - np.mean(X2)) / (np.max(X2) - np.min(X2))

def reduce_upper_outliers(X, reduce=0.01, half_width=4):
    '''
    Since we are looking at dips in the data, we should remove upper outliers.
    The function is taken from here:
    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration

    :returns
    the signal with reduced upper outliers
    '''
    length = len(X[0])
    remove = int(length * reduce) #how many points to reduce
    for x in X:
        sorted_values = np.flip(np.argsort(x)) #store the indices that would sort the array in descending order
        for j in range(remove):
            idx = sorted_values[j] #index to reduce
            new_val = 0
            count = 0
            for k in range(2 * half_width + 1):
                idx2 = idx + k - half_width
                if idx2 < 1 or idx2 >= length or idx == idx2: #to avoid array out of bounds error
                    continue
                new_val += x[idx2]

                count += 1
            new_val /= count  # count will always be positive here
            if new_val < x[idx]:  # just in case there's a few persistently high adjacent values
                x[idx] = new_val
    return X



