import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import pandas as pd
import itertools
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter, defaultdict
import time
import random
import math


def synonyms_and_parts_of_speech(word):
    wordnet_synsets = wordnet.synsets(word)
    sense_freq_dict = defaultdict(float)

    if (len(wordnet_synsets) != 0):
        for synonym in wordnet_synsets:
            for lemma in synonym.lemmas():
                sense_freq_dict[synonym.pos()] = sense_freq_dict[synonym.pos()] + lemma.count() + 1e-3
        return max(sense_freq_dict, key = sense_freq_dict.get).upper()


def generate_random_samples(size, length, vocab_to_int, all_words_train):
    # true targets are filtered here as size of vocab is too small
    # all_words_train = [x if x in vocab_to_int else 'U' for x in all_words_train]
    neg_samples = list()
    # for i in range(size * 5):
    for i in range(size):
        if i % 1000000 == 0:
            pass
            # print(i)
        frs = random.sample(range(length), 1)[0]
        sec = random.sample(range(length), 1)[0]
        if abs(frs - sec) > 2 or (frs == sec):
            fr = vocab_to_int.get(all_words_train[frs], len(vocab_to_int))
            se = vocab_to_int.get(all_words_train[sec], len(vocab_to_int))
            neg_samples.append((fr, se, 0))
    return neg_samples


def loadData():
    stop = stopwords.words("english")

    data_path = 'drive/My Drive/NLP Assignments/A2/data/training/training-data.1m'
    dev_data_path = 'drive/My Drive/NLP Assignments/A2/data/similarity/dev_x.csv'
    test_data_path = 'drive/My Drive/NLP Assignments/A2/data/similarity/test_x.csv'
    more_sentences_data_path = 'drive/My Drive/NLP Assignments/A2/data/training/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'

    dataframe = pd.read_csv(data_path, sep='\t', header=None)
    dev_dataframe = pd.read_csv(dev_data_path).drop(columns='id')
    test_dataframe = pd.read_csv(test_data_path).drop(columns='id')

    refined_data_path = more_sentences_data_path + '/news.en-0000' + str(1) + '-of-00100'
    more_sent_df = pd.read_csv(refined_data_path, sep='\t', nrows=5000, header=None)

    for i in range(2, 10):
        refined_data_path = more_sentences_data_path + '/news.en-0000' + str(i) + '-of-00100'
        temp_df = pd.read_csv(refined_data_path, sep='\t', nrows=5000, header=None)
        more_sent_df = more_sent_df.append(temp_df)

    for j in range(10, 99):
        refined_data_path_2 = more_sentences_data_path + '/news.en-000' + str(j) + '-of-00100'
        temp_df_2 = pd.read_csv(refined_data_path, sep='\t', nrows=5000, header=None)
        more_sent_df = more_sent_df.append(temp_df_2)

    dataframe = dataframe.append(more_sent_df)

    # if nRows > dataframe.size:
    # nRows = dataframe.size

    # dataframe = dataframe.loc[:nRows, ]
    # dev_dataframe = dev_dataframe.loc[:nRows, ]
    # test_dataframe = test_dataframe.loc[:nRows, ]
    # print(dataframe.shape)
    # apply stemming, punctation removal, stopword removal

    dataframe[0] = dataframe[0].apply(
        lambda x: [word for word in word_tokenize(x.lower()) if (word not in stop and word.isalpha())])

    # get all words in training, dev and test sets
    all_words_train = list(itertools.chain(*dataframe[0].values))

    all_words_dev = dev_dataframe["word1"].tolist()
    all_words_dev.extend(dev_dataframe["word2"].tolist())
    all_words_test = test_dataframe["word1"].tolist()
    all_words_test.extend(test_dataframe["word2"].tolist())

    all_sentences_train = np.asarray(dataframe[0])

    all_words_train.extend(all_words_dev)
    all_words_train.extend(all_words_test)

    # get counts for training words
    corpus_counts = Counter(all_words_train)

    # vocab = list(corpus_counts.keys())
    filtered_vocab_list = [item for item, count in corpus_counts.items() if count > 1]
    filtered_vocab_list.append('A')
    filtered_vocab_list.append('N')
    filtered_vocab_list.append('R')
    filtered_vocab_list.append('S')
    filtered_vocab_list.append('V')
    filtered_vocab_list.append('U')
    # print(round(len(vocab)*(1-trim)))

    # take top (1-trim)% of your vocab in frequency descending order
    # vocab = vocab[:round(len(vocab) * (1 - trim))]

    # add RARE keyword to stand in for trimmed words
    # testdev_words = all_words_dev
    # testdev_words.extend(all_words_test)
    # print(testdev_words)

    # unknown_words = list(set(testdev_words) - set(filtered_vocab_list))
    # print('Number of unknown words in dev and test: {}'.format(len(unknown_words)))
    # create dictionary for index of each word
    vocab_to_int = {filtered_vocab_list[i]: i for i in
                    range(0, len(filtered_vocab_list))}  # sort before assigning each word a number?
    # special stopword

    all_contexts_train = list()

    for tokens in all_sentences_train:
        for i in range(0, len(tokens) - 2):
            ind1 = vocab_to_int.get(tokens[i]) if tokens[i] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[i]), vocab_to_int.get("U"))
            ind2 = vocab_to_int.get(tokens[i + 2]) if tokens[i + 2] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[i + 2]), vocab_to_int.get("U"))
            all_contexts_train.append((ind1, ind2, 1))
        for j in range(0, len(tokens) - 1):
            ind1 = vocab_to_int.get(tokens[j]) if tokens[j] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[j]), vocab_to_int.get("U"))
            ind2 = vocab_to_int.get(tokens[j + 1]) if tokens[j + 1] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[j + 1]), vocab_to_int.get("U"))
            all_contexts_train.append((ind1, ind2, 1))
        for k in range(1, len(tokens)):
            ind1 = vocab_to_int.get(tokens[k]) if tokens[k] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[k]), vocab_to_int.get("U"))
            ind2 = vocab_to_int.get(tokens[k - 1]) if tokens[k - 1] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[k - 1]), vocab_to_int.get("U"))
            all_contexts_train.append((ind1, ind2, 1))
        for l in range(2, len(tokens)):
            ind1 = vocab_to_int.get(tokens[l]) if tokens[l] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[l]), vocab_to_int.get("U"))
            ind2 = vocab_to_int.get(tokens[l - 2]) if tokens[l - 2] in vocab_to_int else vocab_to_int.get(synonyms_and_parts_of_speech(tokens[l - 2]), vocab_to_int.get("U"))
            all_contexts_train.append((ind1, ind2, 1))

    print("pos context length: ", len(all_contexts_train))
    neg = generate_random_samples(len(all_contexts_train), len(all_words_train), vocab_to_int, all_words_train)

    all_contexts_train.extend(neg)
    random.shuffle(all_contexts_train)

    return all_contexts_train, filtered_vocab_list, vocab_to_int