import pandas as pd
import re

from collections import Counter
import random

import pickle
from sklearn import svm


class StrategyClassifierTrainer:

    def train_svm(self, input_vectors: list, save_model: bool):
        total = len(input_vectors)
        random.shuffle(input_vectors)
        train_samples = int(total * 0.8)

        train = input_vectors[:train_samples]
        test = input_vectors[train_samples:]

        y = [int(r[0]) for r in train]
        X = [r[1:][0] for r in train]

        clf = svm.SVC(kernel='poly', degree=14).fit(X, y)
        if save_model:
            pickle.dump(clf, open(save_model, 'wb'))

        dev_out, dev_in = [], []

        for sample_x in test:
            if int(sample_x[0]) == 0 and random.random() < 0.7:
                continue
            dev_out.append(int(sample_x[0]))
            Xx = sample_x[1:]
            dev_in.append(Xx)
        print(Counter(dev_out))

    def train_strategy(self, df, strategy: str):
        input_vectors = []
        for index, row in df.iterrows():
            input_vector = list([row[strategy]])
            input_vector.append([row[column] for column in df.columns if re.match("(target\\..+|se\\..+)", column)])
            input_vectors.append(input_vector)

        self.train_svm(input_vectors, False)