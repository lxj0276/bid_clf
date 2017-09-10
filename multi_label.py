import jieba
from psql_manage import Psql
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np



# 1.获取数据
#######################
# 加载停用词词典
def load_stopwords():
    input = open('chinese_stopword.txt', 'r', encoding='utf-8')
    lines = []
    for line in input.readlines():
        lines.append(line.strip())
    return lines


########################
def get_data(psql):
    rows = psql.execute_sql("select id,title,labels from train;")
    ids = []
    titles = []
    for row in rows:
        id = row[0]
        title = row[1].strip()
        ids.append(id)
        titles.append(title)
    return ids, titles


def seg(titles):
    """
    分词
    :param titles:
    :return:
    """
    stop_words = load_stopwords()
    corpus = []
    for title in titles:
        seg_list = []
        for word in jieba.cut(title):
            if word in stop_words:
                # print(word)
                continue
            seg_list.append(word)
        # print " /".join(seg_list)
        corpus.append(' '.join(seg_list))
    return corpus


def insert_seg(psql, ids, corpus):
    psql.execute_commit("alter table train add segment TEXT;")
    print("add table column success")

    try:
        for i in range(len(ids)):
            id = ids[i]
            segment = corpus[i]
            sql = "update train set segment = '" + segment + "' where id = " + str(id)
            psql.cur.execute(sql)
        psql.conn.commit()
    except Exception as e:
        print(e)
        psql.conn.rollback()
    pass


def get_seg(psql):
    rows = psql.execute_sql("select id,segment,labels from train;")
    ids = []
    segments = []
    labels = []
    for row in rows:
        id = row[0]
        segment = row[1].strip()
        label = row[2].strip().split(',')
        ids.append(id)
        segments.append(segment)
        labels.append(label)
    return ids, segments, labels


# 2.数据处理：
# 2.1空间向量转换
# 2.2特征选择
vectorizer = None


def doc_to_vector(X_train, X_test):
    t0 = time()
    global vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("doc_to_vector done in %fs" % (duration))
    print("train:n_samples: %d, n_features: %d" % X_train.shape)
    print("test:n_samples: %d, n_features: %d" % X_test.shape)
    return X_train, X_test


ch2 = None


def select_feature(feature_num, X_train, y_train, X_test):
    t0 = time()
    global ch2
    ch2 = SelectKBest(chi2, k=feature_num)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("select_feature done in %fs" % (time() - t0))
    return X_train, X_test


# 3.分类
def linear_svc_classifier():
    #clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),('classification', LinearSVC(penalty="l2"))])
    #clf = Pipeline([('classification', LinearSVC(penalty="l2"))])
    clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
    return clf
def svc():
    clf = SVC()
    return clf

def nb_classifier():
    clf = MultinomialNB(alpha=.01)
    return clf

# 4.评估
# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    # 训练
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    # 测试集预测
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    # 评估
    print(y_test)
    print(pred)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    print(clf_descr, score, train_time, test_time)


# 5.预测新样本
# 获取titles数据
def get_predict_data(psql):
    datas = psql.get_titles_data()
    ids = []
    titles = []
    for key in datas.keys():
        ids.append(key)
        titles.append(datas[key])
    return ids, titles


def transform_test(predict_data):
    """
    对新数据进行空间向量转换和特征选择
    :param test_data:
    :return: 特征选择后的数据
    """
    global vectorizer
    test_data_vsm = vectorizer.transform(predict_data)
    global ch2
    test_sf = ch2.transform(test_data_vsm)
    return test_sf


def predict_titles(psql,clf):
    ids, titles = get_predict_data(psql)
    corpus = seg(titles)
    test_sf = transform_test(corpus)
    pred = clf.predict(test_sf)
    print(pred)
    for i in range(len(ids)):
        label = list(mlb.classes_[np.where(pred[i, :] == 1)[0]])
        label = " ".join(map(str, label))
        if label == "":  # if the label is empty
            label = "**********"
        print(titles[i],label)


if __name__ == '__main__':
    pass
    psql = Psql()
    # ids, titles = get_data(psql)
    # corpus = seg(titles)
    # insert_seg(psql,ids,corpus)
    # psql.close()


    ids, segments, labels = get_seg(psql)
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=0)
    X_train, X_test = doc_to_vector(X_train, X_test)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.fit_transform(y_test)#将y_test转换为矩阵
    X_train, X_test = select_feature(7000, X_train, y_train, X_test)
    # # 训练
    clf = OneVsRestClassifier(linear_svc_classifier(), n_jobs=-1)
    #clf = OneVsRestClassifier(nb_classifier(), n_jobs=-1)
    #clf = OneVsRestClassifier(svc())

    clf.fit(X_train, y_train)
    #测试集测试
    pred = clf.predict(X_test)
    print(np.mean(pred == y_test))

    # for i in range(pred.shape[0]):
    #     label = list(mlb.classes_[np.where(pred[i, :] == 1)[0]])
    #     label = " ".join(map(str, label))
    #     if label == "":  # if the label is empty
    #         label = "103"
    #     print( str(i) + "," + label + "\n")

    #benchmark(clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    predict_titles(psql, clf)

