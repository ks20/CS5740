import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import featurize
import numpy as np


class Net(nn.Module):
    def __init__(self, V, batch_size):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.vocab = V
        self.dimEmbedding = 50
        self.words = nn.Embedding(len(V) * 3 + 1, self.dimEmbedding)
        self.words.weight.data.uniform_(-0.1, 0.1)
        self.contexts = nn.Embedding(len(V) * 3 + 1, self.dimEmbedding)
        self.contexts.weight.data.uniform_(-0.1, 0.1)
        self.batch_size = batch_size
        self.sigmoid = nn.Sigmoid()
        self.parameters = list(self.words.parameters()) + list(self.contexts.parameters())

    def forward(self, i_w, i_c):
        # then you have to evaluate the sigmoid of the negative weight vec dot context vec
        w = self.words(i_w).view(i_w.shape[0], 1, self.dimEmbedding)
        c = self.contexts(i_c).view(i_w.shape[0], self.dimEmbedding, 1)
        x = torch.bmm(w, c)
        return x

    def write_embeddings(self, filename, unknown_words):
        # get embedding vectors for neural network
        emb_vecs = self.words.weight.data.numpy()
        rare_vec = emb_vecs[len(vocab) - 1]
        # mean_vec = np.mean(emb_vecs, axis = 0).tolist()
        # fill rows for unknown words with average
        unkn_vecs = np.asarray([rare_vec for i in range(len(unknown_words))])

        # combine words with vectors
        unkn_vecs = np.concatenate((np.reshape(unknown_words, (len(unknown_words), 1)), unkn_vecs), axis=1)
        emb_vecs = np.concatenate((np.reshape(self.vocab, (len(self.vocab), 1)), emb_vecs), axis=1)

        # concatenate known and unknown
        all_vecs = np.concatenate((emb_vecs, unkn_vecs), axis=0)
        # print(all_vecs)
        np.savetxt(filename, X=all_vecs, delimiter=' ', fmt='%s', encoding="utf-8")


if __name__ == '__main__':
    print('Loading data...')
    # trims = [0.1, 0.25]
    # batches = [128, 128]
    # tSizes = [5000, 5000]

    '''
    # implement ablations later
    for trim, batch_size, train_size in zip(trims, batches, tSizes):
        print('Training embeddings with {} trimmed data, batch size = {}, and {} training data.'\
            .format(trim, batch_size, train_size))
    print('end ablations.')
    '''

    BATCHSIZE = 50000
    data, vocab, corpus_dict = featurize.loadData()

    # batch_size = 128
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCHSIZE, shuffle=False, num_workers=0)

    print('Data loaded.')
    # print(vocab)
    # print(len(vocab))
    # define the network
    net = Net(vocab, BATCHSIZE)
    # write embeddings to file
    # net.write_embeddings('embeddings_pre.txt', unknown_words)

    # then you need to provide you loss function
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters, lr=0.01)

    for epoch in range(2):
        running_loss = 0
        for i, dat in enumerate(dataloader):
            w, c, y = dat

            # zero the parameter gradients
            y_star = net.forward(w, c)
            optimizer.zero_grad()
            loss = criterion(y_star.type(torch.FloatTensor), y.type(torch.FloatTensor).view(y_star.shape[0], 1,
                                                                                            1))  ## need to update to reflect dimSize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    # net.write_embeddings('embeddings.txt', unknown_words)
    '''
    To evaluate:
    python similarity.py > predictions.csv --e embeddings.txt --w ../data/similarity/dev_x.csv
    python evaluate.py --p predictions.csv --d ../data/similarity/dev_y.csv
    To submit:
    python similarity.py > ../results/test_scores.csv --e embeddings.txt --w ../data/similarity/test_x.csv
    '''