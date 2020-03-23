from nltk import word_tokenize
import re
import numpy as np

def newsgroup_featurize(input_data, featType = 'bow', rem_punct = False, rem_stopwords = False, alphanum_only = False):
    """ Featurizes an input for the newsgroup domain.
    
    Inputs:
        input_data: The input data.
        featType: type of featurization to comppute
        rem_punct: remove punctuation from the bag Y/N
        rem_stopwords: remove stopwords Y/N
        alphanum_only: Whether to remove all characters other than alphanumeric

    Returns:
        features - list of features represented as dictionaries
        vocab - map of full feature space represented as a dictionary
        featuresList - features in list format

    """
    # TODO: Implement featurization of input.
    if (featType == 'bow'):
        vocab = {}
        #create vocab dict of all observed words
        # print(input_data)
        for email in input_data:
            for word in word_tokenize(email):
                word = word.lower()
                if (alphanum_only):
                    word = re.sub('[\W]+', '', word)   
                if (rem_stopwords) and (word in ['the','a','is','was','to']):
                    continue
                if (rem_punct) and (word in ['.',',','?','!']):
                    continue
                # print(word)
                if word in vocab: vocab[word] += 1 
                else: vocab[word] = 1
        # print("Vocab length: " + str(len(vocab)))
        #then go through features again, create a dict for each row of input data
        features = []
        featuresLst = []
        for email in input_data:
            feat = {}
            featLst = []
            for word in word_tokenize(email):
                word = word.lower()
                if word in vocab:
                    feat[word] = 1
                    featLst.append(word)
            features.append(feat)
            featuresLst.append(featLst)
    
    if (featType == 'bigrams'):
        vocab = {}
        #create vocab dict of all observed words
        # print(input_data)
        for email in input_data:
            unigrams = word_tokenize(email)
            unigrams = [x.lower() for x in unigrams]
            for bigram in zip(unigrams, unigrams[1:]):
                if bigram in vocab: 
                    vocab[bigram] += 1
                else:
                    vocab[bigram] = 1
        # print("Vocab length: " + str(len(vocab)))
        features = []
        featuresLst = []
        for email in input_data:
            feat = {}
            featLst = []
            unigrams = word_tokenize(email)
            unigrams = [x.lower() for x in unigrams]
            for bigram in zip(unigrams, unigrams[1:]):
                # print(bigram)
                if bigram in vocab:
                    feat[bigram] = 1 
                    featLst.append(bigram)
            features.append(feat)
            featuresLst.append(featLst)
    return features, vocab, featuresLst 

def newsgroup_label_to_id(input_labels):
    class_to_id = {'comp.graphics': 0, 'comp.os.ms-windows.misc': 1, 'comp.sys.ibm.pc.hardware': 2, 'comp.sys.mac.hardware': 3, 'comp.windows.x': 4,\
        'misc.forsale': 5, 'rec.autos': 6, 'rec.motorcycles': 7, 'rec.sport.baseball': 8, 'rec.sport.hockey': 9,\
        'talk.politics.misc': 10, 'talk.politics.guns': 11, 'talk.politics.mideast': 12, 'sci.crypt': 13, 'sci.electronics': 14,\
        'sci.med': 15, 'sci.space': 16, 'talk.religion.misc': 17, 'alt.atheism': 18, 'soc.religion.christian': 19}
    converted_input_labels = list()
    for i in range(len(input_labels)):
        converted_input_labels.append(class_to_id[input_labels[i]])
    return np.array(converted_input_labels)

def newsgroup_id_to_label(input_labels):
    class_to_id =  {0 : 'comp.graphics', 1:'comp.os.ms-windows.misc', 2:'comp.sys.ibm.pc.hardware', 3:'comp.sys.mac.hardware', 4:'comp.windows.x',\
        5:'misc.forsale', 6 : 'rec.autos', 7:'rec.motorcycles', 8 : 'rec.sport.baseball', 9 : 'rec.sport.hockey',\
        10 :'talk.politics.misc', 11:'talk.politics.guns', 12: 'talk.politics.mideast', 13 : 'sci.crypt', 14 : 'sci.electronics',\
        15 : 'sci.med', 16 : 'sci.space', 17 : 'talk.religion.misc', 18 : 'alt.atheism', 19 : 'soc.religion.christian'}
    converted_input_labels = list()
    for i in range(len(input_labels)):
        converted_input_labels.append(class_to_id[input_labels[i]])
    return np.array(converted_input_labels)

def newsgroup_data_loader(train_data_filename,
                          train_labels_filename,
                          dev_data_filename,
                          dev_labels_filename,
                          test_data_filename):
    """ Loads the data.

    Inputs:
        train_data_filename (str): The filename of the training data.
        train_labels_filename (str): The filename of the training labels.
        dev_data_filename (str): The filename of the development data.
        dev_labels_filename (str): The filename of the development labels.
        test_data_filename (str): The filename of the test data.

    Returns:
        Training, dev, and test data, all represented as a list of (input, label) format.
            for test data, put in some dummy value as the label.
        
        "vocab" dictionary representing the featurization of the data

        list of classes found in the training data. 
    """
    # TODO: Load the data from the text format.
    import numpy as np
    from pandas import read_csv
    
    #read data
    train = read_csv(train_data_filename)['text'].tolist()
    trainlab = read_csv(train_labels_filename)['newsgroup'].tolist()
    dev = read_csv(dev_data_filename)['text'].tolist()
    devlab = read_csv(dev_labels_filename)['newsgroup'].tolist()
    test = read_csv(test_data_filename)['text'].tolist()
    testlab = ['']*len(test)
    
    classes = list(set(trainlab))

    return train, trainlab, dev, devlab, test, testlab, classes
    