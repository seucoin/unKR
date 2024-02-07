import torch.nn as nn
import torch
from .model import Model
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import torch.optim as optim
import numpy as np
import multiprocessing
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class LSTMModel(nn.Module):
    """
    Create LSTM model.
    """

    def __init__(self, dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(dim, hidden_size=dim, batch_first=True)
        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.fc(lstm_output[:, -1, :])
        output = self.sigmoid(output)
        return output   # The shape is [bs, 1]，and every elment represent the probability of the corresponding sample's result, within range [0,1]


class UKGsE(Model):

    def __init__(self, args):
        super(UKGsE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.lstm_model = LSTMModel(args.emb_dim).to(args.gpu)

        self.init_emb()

    def init_emb(self):
        """
        Pre-train the entity and relation embeddings using Word2Vec model.
        """
        self.ready_data()

        corpus = self.args.data_path + '/train.tsv.txt'    # relation is represented in the form of r+relation_id

        word2vec_model = Word2Vec(LineSentence(corpus), vector_size=int(self.args.emb_dim), window=2, sample=0,
                                  epochs=10, negative=5, min_count=1,
                                  sg=self.args.sg,  # skip_gram (1) or CBOW (0)
                                  workers=multiprocessing.cpu_count())
        word2vec_model.init_sims(replace=True)
        # word2vec_model.wv.save_word2vec_format(vectors)

        # Initialize the embedding layer of entity and relation
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)

        # Assign the embeddings of entity and relation using the word vectors of Word2Vec model
        for i in range(self.args.num_ent):
            ent_word = str(i)  # Construct the corresponding word according to the entity's id
            if ent_word in word2vec_model.wv:
                ent_vector = torch.Tensor(word2vec_model.wv[ent_word].copy()).unsqueeze(0)
                self.ent_emb.weight.data[i] = ent_vector  # Assign word vectors to numbered entity vectors

        for i in range(self.args.num_rel):
            rel_word = "r" + str(i)  # Construct the corresponding word according to the relation's id
            if rel_word in word2vec_model.wv:
                rel_vector = torch.Tensor(word2vec_model.wv[rel_word].copy()).unsqueeze(0)
                self.rel_emb.weight.data[i] = rel_vector

        self.ent_emb = self.ent_emb.to(self.args.gpu)
        self.rel_emb = self.rel_emb.to(self.args.gpu)

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        
        if mode == 'head_predict' and head_emb.shape[0] == 1:
            # print('head_predict')
            head_emb = head_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            X = torch.empty(head_emb.shape[0] * tail_emb.shape[0], 3, tail_emb.shape[1])  # [bs*num_ent, 3, dim]

            tail_emb_repeat = tail_emb.repeat_interleave(head_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb.repeat_interleave(head_emb.shape[0], dim=0)

            head_emb_repeat = head_emb.repeat(tail_emb.shape[0], 1)
            head_emb_repeat = head_emb_repeat.view(-1, head_emb.shape[1])

            X[:, 0, :] = head_emb_repeat
            X[:, 1, :] = relation_emb_repeat
            X[:, 2, :] = tail_emb_repeat
            # print(X.shape)

        elif mode == 'tail_predict' and tail_emb.shape[0] == 1:  # tail_predict
            # print('tail_predict')
            head_emb = head_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]

            X = torch.empty(head_emb.shape[0] * tail_emb.shape[0], 3, head_emb.shape[1])

            head_emb_repeat = head_emb.repeat_interleave(tail_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb.repeat_interleave(tail_emb.shape[0], dim=0)

            tail_emb_repeat = tail_emb.repeat(head_emb.shape[0], 1)
            tail_emb_repeat = tail_emb_repeat.view(-1, tail_emb.shape[1])

            X[:, 0, :] = head_emb_repeat
            X[:, 1, :] = relation_emb_repeat
            X[:, 2, :] = tail_emb_repeat
            # print(X.shape)

        else:
            head_emb = head_emb.squeeze(1)
            relation_emb = relation_emb.squeeze(1)
            tail_emb = tail_emb.squeeze(1)  # [bs, dim]
            X = torch.stack((head_emb, relation_emb, tail_emb), dim=1)

        X = X.to(self.args.gpu)
        y = self.lstm_model(X)
        # print(y.shape)
        score = y.to(self.args.gpu)
        batch_size = relation_emb.shape[0]
        score = score.reshape(batch_size, -1)  

        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        # self.lstm()

        triples = triples[:, :3].to(torch.int)
        # c = triples[:, 3]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)  # [bs, 1, dim]
        head_emb = head_emb.squeeze(1)
        relation_emb = relation_emb.squeeze(1)
        tail_emb = tail_emb.squeeze(1)   # [bs, dim]

        X_train = torch.stack((head_emb, relation_emb, tail_emb), dim=1).to(self.args.gpu)  # [bs, 3, dim]

        score = self.lstm_model(X_train)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)
        triples = triples.to(self.args.gpu)

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def ready_data(self):
        """
        The relationships and entities in the file are numbered with numbers, and there are duplicates, so the relationship ID is preceded by the prefix 'r'
        """
        base = self.args.data_path

        train_file = base + '/train.tsv'
        test_file = base + '/test.tsv'

        train_with_confidence_file = open(train_file + ".txt", 'w')
        test_with_confidence_file = open(test_file + ".txt", 'w')

        with open(train_file, 'r') as f1, open(test_file, 'r') as f2:
            lines = f1.readlines()
            for line in lines:
                triplet = line.strip().split()
                train_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                                 + triplet[2] + '\n')  # for step_1

            lines = f2.readlines()
            for line in lines:
                triplet = line.strip().split()
                test_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                                + triplet[2] + '\t' + triplet[3] + '\n')

        train_with_confidence_file.close()
        test_with_confidence_file.close()

