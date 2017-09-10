import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from NLP_options import train_data_to_vsm, train_feature_selection, test_data_to_vsm, \
    process_multi_label_feature_selection, test_feature_selection,get_linearSVC
from sklearn.metrics import jaccard_similarity_score
# 获取数据,数据预处理
from psql_manage import Psql
import jieba


#######################
def load_stopwords():
    input = open('chinese_stopword.txt', 'r', encoding='utf-8')
    lines = []
    for line in input.readlines():
        lines.append(line.strip())
    return lines


########################
stop_words = load_stopwords()
psql = Psql()
ids, titles, labels = psql.get_train_data()
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

# VSM
#
X_train, y_train = corpus[6000:], labels[6000:]
process_multi_label_feature_selection(selected_num=3000, corpus=X_train, labels=y_train)
# X_train = test_data_to_vsm(X_train)
# X_train = test_feature_selection(X_train)
#
# X_test, y_test = corpus[:6000], labels[:6000]
# X_test_vsm = test_data_to_vsm(X_test)
# X_test_selected = test_feature_selection(X_test_vsm)
#
# # transform y into a matrix 模型拟合及预测
# mb = MultiLabelBinarizer()
# y_test = mb.fit(y_train).transform(y_test)
# y_train = mb.fit_transform(y_train)
#
# # fit the model and predict 模型评估
# # clf = OneVsRestClassifier(SVC(kernel='linear'))
# # clf = OneVsRestClassifier(MultinomialNB(alpha=.01))
# clf = OneVsRestClassifier(get_linearSVC())
#
#
# clf.fit(X_train, y_train)
# pred_y = clf.predict(X_test_selected)
#
# # training set result
# # y_predicted = clf.predict(X_train)
# ovr_jaccard_score = jaccard_similarity_score(y_test, pred_y)
# print(ovr_jaccard_score)
# # report
# print(metrics.classification_report(y_test, pred_y))
#
# print(np.mean(pred_y == y_test))
