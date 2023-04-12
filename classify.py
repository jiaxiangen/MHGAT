from __future__ import print_function


import numpy
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)

def accuracy(logits, labels):
    labels = numpy.asarray(labels).reshape(-1)
    indices = numpy.argmax(logits, axis=1)
    correct = numpy.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

class Classifier(object):

    def __init__(self, clf):
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = X
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y, prints=True):
        top_k_list = [len(l) for l in Y]
        results = {}
        Y_ = self.predict(X, top_k_list)
        # results['acc'] = accuracy(Y_,Y)
        Y = self.binarizer.transform(Y)
        # results['accuracy'] = accuracy_score(Y, Y_)
        # results['precision'] = precision_score(Y, Y_, average='micro')
        # results['recall'] = recall_score(Y, Y_, average='micro')
        # results['f1-score'] = f1_score(Y, Y_, average='micro')
        results['auc'] = roc_auc_score(Y.todense(), Y_, average='micro')
        averages = ["micro", "macro", "samples", "weighted"]
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        if prints:
            print('-------------------')
            print(results)
        return results

    def predict(self, X, top_k_list):
        X_ = X
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent=0.2, seed=0, prints=True):
        state = numpy.random.get_state()
        Y = Y.reshape(-1,1)

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = X[shuffle_indices[:training_size]]
        Y_train = Y[shuffle_indices[:training_size]]
        X_test = X[shuffle_indices[training_size:]]
        Y_test = Y[shuffle_indices[training_size:]]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test, prints)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1])
    fin.close()
    return X, Y
