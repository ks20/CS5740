""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
from propername import *
from newsgroup import *
from util import save_results


import dynet_config
import dynet as dy
dynet_config.set(random_seed=0)
from util import evaluate, load_data
dy.renew_cg()

class MultilayerPerceptronModel():
    """ Maximum entropy model for classification.

    Attributes: 
        nn
        trainer:
        corpus: all posssible features used in the nn
        dim: 
        hidden
        vocab_size: size of the corpus
        numClasses: number of classes to predict


    """
    def __init__(self, m, corpus, numClasses, in_dim):
        self.nn = m
        self.trainer = dy.AdamTrainer(self.nn)
        self.corpus = corpus

        self.dim = 10
        self.in_dim = in_dim
        self.hidden = 250
        self.vocab_size = len(self.corpus)
        self.numClasses = numClasses

        self._pW1 = self.nn.add_parameters((self.dim, self.hidden), init="glorot")
        #self._pW2 = self.nn.add_parameters((self.in_dim, self.hidden), init="glorot")
        self._pB1 = self.nn.add_parameters((1, self.hidden), init=0)
        self._kW1 = self.nn.add_parameters((self.hidden, self.numClasses), init="glorot")
        self._kB1 = self.nn.add_parameters((1, self.numClasses), init=0)
        self.lookup = self.nn.add_lookup_parameters((200000, self.dim), init="glorot")

    def train(self, input_data):
        accuracy_per_epoch, loss_per_epoch = list(), list()
        for epoch in range(5):
            print("EPOCH:", epoch)
            train_loss = 0
            train_correct = 0

            for x, y in input_data:
                if (not x):
                    continue
                loss, bestGuess = self.create_network_return_loss(x, y)

                if bestGuess == y:
                    train_correct = train_correct + 1

                train_loss = train_loss + loss.value()
                loss.backward()
                self.trainer.update()

            print(train_loss / len(input_data))
            accuracy_per_epoch.append(train_correct / len(input_data))
            loss_per_epoch.append(train_loss / len(input_data))

    #Method Implementation courtesy of: https://dynet.readthedocs.io/en/latest/tutorials_notebooks/API.html
    def create_network_return_loss(self, x, label):
        dy.renew_cg()
        emb_vectors = [self.lookup[self.corpus.get(item, len(self.corpus))] for item in x]
        calc_avg = dy.average(emb_vectors)
        emb_vectors_mean = dy.reshape(calc_avg, (1, self.dim))
        z1 = (emb_vectors_mean * self._pW1) + self._pB1
        a1 = dy.tanh(z1)
        net_output = dy.softmax(dy.reshape((a1 * self._kW1) + self._kB1, (self.numClasses,)))
        loss = -dy.log(dy.pick(net_output, label))
        bestGuess = np.argmax(net_output.npvalue())
        return loss, bestGuess

    #Method Implementation courtesy of: https://dynet.readthedocs.io/en/latest/tutorials_notebooks/API.html
    def create_network_return_best(self, x):
        dy.renew_cg()
        emb_vectors = [self.lookup[self.corpus.get(item, len(self.corpus))] for item in x]
        calc_avg = dy.average(emb_vectors)
        emb_vectors_mean = dy.reshape(calc_avg, (1, self.dim))
        z1 = (emb_vectors_mean * self._pW1) + self._pB1
        a1 = dy.tanh(z1)
        net_output = dy.softmax(dy.reshape((a1 * self._kW1) + self._kB1, (self.numClasses,)))
        return np.argmax(net_output.npvalue())

    def predict(self, model_input):
        predictions = list()
        for pName in model_input:
            pred = self.create_network_return_best(pName)
            predictions.append(pred)
        return predictions

if __name__ == "__main__":
    train_data_file_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/train/train_data.csv'
    train_labels_file_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/train/train_labels.csv'
    dev_data_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/dev/dev_data.csv'
    dev_labels_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/dev/dev_labels.csv'
    test_data_filename_pNouns = '~/Downloads/assignment-1-fixed-kushandjay-master/data/propernames/test/test_data.csv'
    
    m = dy.ParameterCollection()
    train_properNouns, dev_data, test_properNouns = propername_data_loader(train_data_file_pNouns, train_labels_file_pNouns, dev_data_filename_pNouns, dev_labels_filename_pNouns, test_data_filename_pNouns)

    X_train, corpus, corpusLst = propername_featurize(train_properNouns[:, 0])
    train_labels = train_properNouns[:, 1]
    Y_train = propername_label_to_id(train_labels)
    training_data = zip(corpusLst, Y_train)

    X_dev, devCorpus, devCorpusLst = propername_featurize(dev_data[:, 0])
    dev_labels = dev_data[:, 1]
    #Y_dev = propername_label_to_id(dev_labels)
    #dev_input_data = zip(devCorpusLst, Y_dev)

    X_test, _, _ = propername_featurize(test_properNouns[:, 0])

    numClasses = len(set(Y_train))
    model = MultilayerPerceptronModel(m, corpus, numClasses, len(X_train))
    model.train(training_data)

    #Y_dev_predicted = model.predict(X_dev)
    #print("DEV PREDICTED: ", Y_dev_predicted)

    Y_test_predicted = model.predict(X_test)
    print("TEST PREDICTED: ", Y_test_predicted)
    
    # Predict on the development set.
    # dev_accuracy = evaluate(model,
    #                         dev_data,
    #                         os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"))

    # # Predict on the test set.
    # # Note: We don't provide labels for test, so the returned value from this
    # # call shouldn't make sense.
    # evaluate(model,
    #          X_test,
    #          os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"))

    ##################################
    # Newsgroups
    ##################################
   
    train_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_data.csv'
    train_labels_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/train/train_labels.csv'
    dev_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_data.csv'
    dev_labels_filename ='~/assignment-1-fixed-kushandjay/data/newsgroups/dev/dev_labels.csv'
    test_data_filename = '~/assignment-1-fixed-kushandjay/data/newsgroups/test/test_data.csv'

    m = dy.ParameterCollection()
    # get raw data and labels for train, dev, and test set
    trainraw, trainlab, devraw, devlab, testraw, testlab, classes = newsgroup_data_loader(train_data_filename, train_labels_filename,dev_data_filename,\
        dev_labels_filename, test_data_filename)

    X_train, corpus, corpusLst = newsgroup_featurize(trainraw, alphanum_only=True)
    Y_train = newsgroup_label_to_id(trainlab)
    training_data = [x for x in zip(corpusLst, Y_train)]

    # X_dev, devCorpus, devCorpusLst = newsgroup_featurize(devraw, alphanum_only=True)
    # Y_dev = newsgroup_label_to_id(devlab)
    # dev_input_data = [x for x in zip(devCorpusLst, Y_dev)]

    X_test, _, _ = newsgroup_featurize(testraw, alphanum_only = True)
    numClasses = len(set(Y_train))

    model = MultilayerPerceptronModel(m, corpus, numClasses, len(X_train))
    print('Training now.')
    model.train(training_data)

    Y_test_predicted = model.predict(X_test)
    Y_test_predicted = newsgroup_id_to_label(Y_test_predicted)
    
    save_results(Y_test_predicted, 'mlp_newsgroup_test_predictions.csv')

    