""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np
import numpy as np
from newsgroup import newsgroup_data_loader, newsgroup_featurize
from propername import propername_data_loader, propername_featurize
from util import evaluate, load_data

class PerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:
    weights - list with len = number of classes. Each entry is a dictionary of all possible features with weights as values
    classes - list of classes in the classification problem

    """
    def __init__(self, dimensions, givenClasses):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        self.classes = givenClasses
        self.weights = []
        self.zero_weights(dimensions)

    def rand_weights(self, dimensions):
        for _ in range(len(self.classes)):
            w_j = {}
            for k in dimensions:
                w_j[k] = np.random.rand()/100
            self.weights.append(w_j)
        # for j in range(len(self.classes)):
            # print(self.weights[j]['revolutionized'])

    
    def zero_weights(self, dimensions):
        for _ in range(len(self.classes)):
            w_j = {}
            for k in dimensions:
                w_j[k] = 0
            self.weights.append(w_j)
    
    def get_weights(self):
        return self.weights

    def get_classes(self):
        return self.classes


    def train(self, training_data, max_epochs, stopping_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
            max_epochs: number of epochs to spend training
            stopping_data: development dataset for evaluating performance at beginining of epoch
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        train_acc = []
        heldout_acc = []
        # remember to implement bias via featuriser
        while (max_epochs > 0):
            # ctr = 0
            print('Epochs left: {}'.format(max_epochs))
            for x, y in training_data:
                yStars = []
                for i in range(len(self.classes)):
                    w_dot_phi = 0
                    for key in x:
                        # print(self.weights[i])
                        # print(x)
                        w_dot_phi += self.weights[i][key] * x[key]
                    yStars.append(w_dot_phi)
                predClass = self.classes[np.argmax(np.asarray(yStars))]
                # if ((ctr <50) or (len(training_data) - ctr < 50)):
                    # print('Y* scores: ')
                    # print(yStars)
                    # print('Predicted class for ({}): {}'.format(y, predClass))
                # update weights
                if (y != predClass):
                    for i, clss in enumerate(self.classes):
                        if clss == y:
                            for key in x:
                                self.weights[i][key] += x[key]
                        else:
                            for key in x:
                                self.weights[i][key] -= x[key]
                # ctr+=1  
            max_epochs -= 1
            res_train, _ = evaluate(self, training_data, '-')
            res_heldout, _ = evaluate(self, stopping_data, '-')
            print('Training accuracy: {}'.format(res_train))
            print('Held-out accuracy: {}'.format(res_heldout))
            train_acc.append(res_train)
            heldout_acc.append(res_heldout)
        # print(self.weights)
        return None

    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a dictionary.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        yStars = []
        for i in range(len(self.classes)):
            w_dot_phi = 0
            for key in model_input:
                # test if input feature is represented in the weights, if not, skip
                if (key in self.weights[i]):
                    # print('Found a key in weights!')
                    w_dot_phi += self.weights[i][key] * model_input[key]
            yStars.append(w_dot_phi)
        # print(yStars)
        predClass = self.classes[np.argmax(np.asarray(yStars))]
        return predClass

if __name__ == "__main__":
    
    ##################################
    # Newsgroups
    ##################################
    print('Building Perceptron with optimal configuration for newsgroup data')
    train_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_data.csv'
    train_labels_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_labels.csv'
    dev_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_data.csv'
    dev_labels_filename ='~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_labels.csv'
    test_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/test/test_data.csv'

    # get raw data and labels for train, dev, and test set
    trainraw, trainlab, devraw, devlab, testraw, testlab, classes = newsgroup_data_loader(train_data_filename, train_labels_filename,dev_data_filename,\
        dev_labels_filename, test_data_filename)

    # trainraw.extend(devraw)
    # trainlab.extend(devlab)

    # featurize the same
    train_feat, vocab, _= newsgroup_featurize(trainraw, featType = 'bow', rem_punct = False, rem_stopwords = False, alphanum_only=True)
    dev_feat, _, _ = newsgroup_featurize(devraw, featType = 'bow', rem_punct = False, rem_stopwords = False, alphanum_only=True)
    # test_feat, _ = newsgroup_featurize(testraw, featType = 'bigrams', rem_punct = False, rem_stopwords = False)

    # print(vocab)
    # reshape datasets as list of tuples for use in Perceptron model
    train = [x for x in zip(train_feat, trainlab)]
    dev = [x for x in zip(dev_feat, devlab)]
    # test = [x for x in zip(test_feat, testlab)]

    perc = PerceptronModel(vocab, classes)
    perc.train(train, 1, dev)
    # res = evaluate(perc, test, 'perceptron_newsgroup_test_predictions.csv')

    
    ######################################
    # Propernames
    ######################################
    
    print('Building Perceptron with optimal configuration for propernames data')
    train_data_filename = '~/assignment-1-fixed-kushandjay/data/propernames/train/train_data.csv'
    train_labels_filename = '~/assignment-1-fixed-kushandjay/data/propernames/train/train_labels.csv'
    dev_data_filename = '~/assignment-1-fixed-kushandjay/data/propernames/dev/dev_data.csv'
    dev_labels_filename ='~/assignment-1-fixed-kushandjay/data/propernames/dev/dev_labels.csv'
    test_data_filename = '~/assignment-1-fixed-kushandjay/data/propernames/test/test_data.csv'

    train, dev, test = propername_data_loader(train_data_filename, train_labels_filename,dev_data_filename,\
        dev_labels_filename, test_data_filename)

    trainraw, trainlab, devraw, devlab, testraw, testlab = train[:,0], train[:,1], dev[:,0], dev[:,1], test[:,0], test[:,1]
    classes = np.unique(train[:,1]).tolist()

    trainraw = trainraw.tolist()
    trainlab = trainlab.tolist()
    devraw = devraw.tolist()
    devlab = devlab.tolist()

    trainraw.extend(devraw)
    trainlab.extend(devlab)

    train_feat, vocab, _ = propername_featurize(trainraw, 3)
    dev_feat, _, _ = propername_featurize(devraw, 3)
    test_feat, _, _ = propername_featurize(testraw, 3)

    #reshape datasets as list of tuples for use in Perceptron model
    train = [x for x in zip(train_feat, trainlab)]
    dev = [x for x in zip(dev_feat, devlab)]
    test = [x for x in zip(test_feat, testlab)]

    # print(vocab)
    perc = PerceptronModel(vocab, classes)
    perc.train(train, 4, dev)
    res = evaluate(perc, test, 'perceptron_propernames_test_predictions.csv')
    
