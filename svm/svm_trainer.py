import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import svm, preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


class SVMTrainer:

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def readData(self, input_file):
        # read csv data
        # input_file = "params/iSTART 4-7 PrePost SEs_FINAL_outv12_0.05000000074505806_0.0.csv"
        output_directory = "params/out/1505/"
        df = pd.read_csv(input_file, header=0, dtype={'a': np.float64}, delimiter=",", low_memory=False)

        return df

    def train_and_select_features(self, df: DataFrame, output_directory: str):
        # filter non-zeros
        dfNonZeros = df[df['Human Score'] != 0]
        # print("Before zeros: ", dfNonZeros.shape)

        # for non-zero do svm classification
        X = dfNonZeros.loc[:, 'Previous text (Wu-Palmer)':]
        y = dfNonZeros["Human Score"]

        # print("Before Nan: ", X.shape)
        imp = SimpleImputer(missing_values='NaN', strategy='mean')
        imp = imp.fit(X)
        X = imp.transform(X)
        # print("After nan: ", X.shape)

        # feature selection
        # 1. Variance threshold
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X = sel.fit_transform(X, y)
        # print(sel.get_support(indices=True))
        # print("VarianceThreshold: ", X.shape)

        # 2. Univariate feature selection
        X = SelectKBest(f_classif, k=300).fit_transform(X, y)
        print("Select k best : ", X.shape, " ", y.shape)
        np.savetxt("DataNotNormalized.csv", X, delimiter=",")

        # normalize data
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        print("Normalized data: ", X.shape)
        np.savetxt("DataNormalized.csv", X, delimiter=",")

        # #eliminate nan and inf
        y = np.nan_to_num(y)
        X = np.nan_to_num(X)

        # 10-fold for testing
        kFold = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

        clf = svm.NuSVC(kernel='rbf', nu=0.1, decision_function_shape='ovr')

        i = 0
        for train_index, test_index in kFold.split(X, y):
            # select fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # print("X Mean and Std for fold: " + str(i) + " " + str(X_test.mean()) + " " + str(X_test.std()))
            # print("Y Mean and Std for fold: " + str(i) + " " + str(y_test.mean()) + " " + str(y_test.std()))

            clf.fit(X_train, y_train)

            scoreTrain = clf.score(X_train, y_train)
            score = clf.score(X_test, y_test)
            # print("SVM training: ", score, " ", scoreTrain)

            y_pred = clf.predict(X_test)
            # adiacent = sum(abs(y_pred - y_test) <= 1)
            print(sum(abs(y_pred - y_test) <= 1) / float(y_test.shape[0]))

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            print("Matrix : ", cnf_matrix)

            # Plot non-normalized confusion matrix
            class_names = ['1', '2', '3']
            plt.figure()
            self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                       title='Confusion matrix, without normalization')
            self.plt.savefig(output_directory + "noNorm" + str(i) + ".png")

            # Plot normalized confusion matrix
            plt.figure()
            self.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                       title='Normalized confusion matrix')

            plt.savefig(output_directory + "Lstm" + str(i) + ".png")
            i = i + 1

        # print("Predictor params: ", clf.get_params())
