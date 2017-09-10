from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import scipy.sparse as sp
import numpy as np

"""
参考 http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

"""


def process_multi_label_feature_selection(corpus, labels, selected_num):
    multi_labels = True
    new_train_x = []
    new_train_y = []
    if multi_labels:
        for i in range(len(labels)):
            x = corpus[i]
            y_list = labels[i]
            for y in y_list:
                new_train_x.append(x)
                new_train_y.append(y)
    data_vsm = train_data_to_vsm(new_train_x)
    train_feature_selection(selected_num, data_vsm, new_train_y)


def train_feature_selection(selected_num, train_x, train_y):
    pass
    global ch2
    t0 = time()
    ch2 = SelectKBest(chi2, k=selected_num)
    X_train = ch2.fit_transform(train_x, train_y)
    print("feature selection done in %fs" % (time() - t0))
    print("样本数目为：%d" % len(train_y))
    print("类别数目为：%d" % len(set(train_y)))
    return X_train


def test_feature_selection(X_test):
    global ch2
    X_test = ch2.transform(X_test)
    return X_test


vectorizer = None
ch2 = None


def train_data_to_vsm(data_train):
    """
    将训练集转换为 空间向量表示 权重采用 tfidf
    :param data_train:
    :return:
    """
    global vectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    print("样本数目: %d, 空间转换后词特征数目: %d" % X_train.shape)
    print('doc data to vectorizer by weight tfidf done. time cost : %f s' % (time() - t0))
    return X_train


def test_data_to_vsm(data_test):
    """
    测试集 空间向量转换
    :param data_test:
    :return:
    """
    t0 = time()
    global vectorizer
    X_test = vectorizer.transform(data_test)
    print("样本数目: %d, 空间转换后词特征数目:%d" % X_test.shape)
    print("测试集空间向量转换耗时： %f s" % (time() - t0))
    return X_test


# #############################################################################
# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


def get_random_forest_clf():
    print("=" * 20)
    print("Random forest")
    return RandomForestClassifier(n_estimators=100)


def get_multionmialNB():
    print("=" * 20)
    print("Naive Bayes")
    return MultinomialNB(alpha=.01)


def get_linearSVC():
    print("=" * 20)
    print("LinearSVC with L1-based feature selection")
    return Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                     ('classification', LinearSVC(penalty="l2"))])
