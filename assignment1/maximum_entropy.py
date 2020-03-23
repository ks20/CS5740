""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np
import scipy.optimize as opt

from util import evaluate, load_data, save_results
from propername import *
from newsgroup import *

class MaximumEntropyModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self, corpus, classes):
        # Initialize the parameters of the model.
        self.corpus = corpus
        self.classes = classes
        self.weights = None
        self.train_loss = list()
        self.train_accuracy = list()
        self.input_data = None

    def train(self, input_data):
        self.input_data = input_data
        self.initialGuess = self.zero_weights(len(self.classes))
        self.minPos = opt.fmin_l_bfgs_b(func=self.fn_log_likelihood,
                                        x0=self.initialGuess,
                                        fprime=self.fn_deriv_log_likelihood, maxiter=4)

        #print("RESULT: ", self.minPos)

    def zero_weights(self, dimensions):
        return np.zeros((4 * (len(corpus) + 1) * dimensions))

    def fn_log_likelihood(self, w_raw):
        self.weights = w_raw.reshape(-1, len(self.classes))
        lossCalc = 0
        numAccurate = 0

        for x, y in self.input_data:
            #print("X: ", x)
            #print("Y: ", y)
            softmax_res = self.feedForward(x)
            best_guess = np.argmax(softmax_res)
            if best_guess == y:
                numAccurate = numAccurate + 1

            lossCalc = lossCalc - np.log(softmax_res[y])

        totalLoss = lossCalc / len(self.input_data)
        accuracy = numAccurate / len(self.input_data)
        self.train_loss.append(totalLoss)
        self.train_accuracy.append(accuracy)

        print("TRAIN LOSS: ", totalLoss)
        return totalLoss

    def fn_deriv_log_likelihood(self, w_raw):
        self.weights = w_raw.reshape(-1, len(self.classes))
        grad = np.zeros(self.weights.shape)
        whitenMat = np.eye(len(self.classes))[Y_train.reshape(-1)]
        res = list(zip(*self.input_data))
        X_train = res[0]

        for x, y in zip(X_train, whitenMat):
            probs = self.feedForward(x)
            delta = (probs - y) / len(X_train)
            for word in x:
                lookupKey = self.corpus.get(word, len(self.corpus))
                grad[lookupKey, :] = grad[lookupKey, :] + delta
        return grad.reshape(w_raw.shape)

    def feedForward(self, x):
        zeroVec = np.zeros(len(self.classes))
        for item in x:
            lookupKey = self.corpus.get(item, len(self.corpus))
            #print("Lookup key: ", lookupKey)
            zeroVec = zeroVec + self.weights[lookupKey, :]
        zeroVec = zeroVec - np.max(zeroVec) #subtract min or max?
        return self.softmax(zeroVec)

    def softmax(self, wf):
        return np.exp(wf) / np.sum(np.exp(wf))

    def predict(self, model_input):
        predictions = list()
        for pName in model_input:
            pred = np.argmax(self.feedForward(pName))
            predictions.append(pred)
        return predictions

if __name__ == "__main__":
    ##################################
    ## Propernames
    ##################################
    train_data_file_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/train/train_data.csv'
    train_labels_file_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/train/train_labels.csv'
    dev_data_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/dev/dev_data.csv'
    dev_labels_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/dev/dev_labels.csv'
    test_data_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/test/test_data.csv'
    
    # Train the model using the training data.
    train_properNouns, dev_data, test_properNouns = propername_data_loader(train_data_file_pNouns, train_labels_file_pNouns, dev_data_filename_pNouns, dev_labels_filename_pNouns, test_data_filename_pNouns)

    X_train, corpus, corpusLst = propername_featurize(train_properNouns[:, 0])
    train_labels = train_properNouns[:, 1]
    Y_train = propername_label_to_id(train_labels)
    training_data = zip(corpusLst, Y_train)

    X_test, _, _ = propername_featurize(test_properNouns[:, 0])

    classes = list(set(train_labels))
    model = MaximumEntropyModel(corpus, classes)
    model.train(training_data)

    Y_test_predicted = model.predict(X_test)
    print("TEST PREDICTED: ", Y_test_predicted)
    
    # Predict on the development set.
    # dev_accuracy = evaluate(model,    
    #                         dev_data,
    #                         os.path.join("results", "maxent_" + data_type + "_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    #evaluate(model,
             #test_data,
             #os.path.join("results", "maxent_" + data_type + "_test_predictions.csv"))

    ###############################
    ## Newsgroups
    ###############################
    
    train_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_data.csv'
    train_labels_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_labels.csv'
    dev_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_data.csv'
    dev_labels_filename ='~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_labels.csv'
    test_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/test/test_data.csv'

    # get raw data and labels for train, dev, and test set
    trainraw, trainlab, devraw, devlab, testraw, testlab, classes = newsgroup_data_loader(train_data_filename, train_labels_filename,dev_data_filename,\
        dev_labels_filename, test_data_filename)
    print('Data loaded.')

    X_train, corpus, corpusLst = newsgroup_featurize(trainraw, alphanum_only=True)
    Y_train = newsgroup_label_to_id(trainlab)
    print('Features calculated.')
    training_data = [x for x in zip(corpusLst, Y_train)]
    #print("Training: ", training_data)
    #print("Corpus List: ", corpusLst)

    X_dev, _, _ = newsgroup_featurize(testraw[:1000], alphanum_only = True)
    Y_dev = newsgroup_label_to_id(devlab[:1000])

    X_test, _, _corpusLstTest = newsgroup_featurize(testraw, alphanum_only = True)

    classes = list(set(Y_train))
    model = MaximumEntropyModel(corpus, classes)
    model.train(training_data)
    Y_pred = model.predict(X_dev)
    print("Y DEV: ", Y_dev[0:10])
    print("Y PRED: ", Y_pred[0:10])
    # print(Y_pred, Y_dev)
    count = 0
    for i in range(len(Y_dev)):
        if Y_pred[i] == Y_dev[i]:
            count += 1
    acc = count/len(Y_dev)
    print(acc)

    Y_test = model.predict(X_test)
    Y_test = newsgroup_id_to_label(Y_test)

    save_results(Y_test, 'maxent_newsgroup_test_predictions.csv')