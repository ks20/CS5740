import numpy as np
from pandas import read_csv

def propername_featurize(input_data, n):
    """ Featurizes an input for the proper name domain.

    Inputs:
        input_data: The input data.
        n: n-grams to consider
    """
    nGramVocab = {}
    nGramVocabLst = []

    for properName in input_data:
        # properName = properName.lower()
        nGramLst = getNgrams(properName, n)
        nGramVocabLst.append(nGramLst)
        for nGram in nGramLst:
            if nGram in nGramVocab:
                nGramVocab[nGram] = nGramVocab[nGram] + 1
            else:
                nGramVocab[nGram] = 1

    features = []
    for properName in input_data:
        feat = {}
        biGramLst = getNgrams(properName, n)
        for biGram in biGramLst:
            feat[biGram] = 1
        features.append(feat)

    return features, nGramVocab, nGramVocabLst

def getNgrams(input_data, n):
    nGrams = [input_data[i: i + n] for i in range(len(input_data) - n+1)]
    return nGrams

def propername_label_to_id(input_labels):
    class_to_id = {'place': 0, 'person': 1, 'drug': 2, 'company': 3, 'movie': 4}
    converted_input_labels = list()
    for i in range(len(input_labels)):
        converted_input_labels.append(class_to_id[input_labels[i]])
    return np.array(converted_input_labels)

def propername_data_loader(train_data_filename,
                           train_labels_filename,
                           dev_data_filename,
                           dev_labels_filename,
                           test_data_filename):

    X_train = read_csv(train_data_filename).to_numpy()
    Y_train = read_csv(train_labels_filename).to_numpy()
    train = np.concatenate((X_train, Y_train), axis = 1)[:,[1,3]]

    X_dev = read_csv(dev_data_filename).to_numpy()
    Y_dev = read_csv(dev_labels_filename).to_numpy()
    dev = np.concatenate((X_dev, Y_dev), axis = 1)[:,[1,3]]

    X_test = read_csv(test_data_filename).to_numpy()
    Y_test = np.empty((X_test.shape[0],1))
    test = np.concatenate((X_test, Y_test), axis = 1)[:,[1,2]]

    return train, dev, test
