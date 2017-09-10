# coding=utf-8
import time
import re
import os
import sys
import codecs
import numpy as np
import scipy
#from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = []

# 读取预料 一行预料为一个文档
for line in open('test_result.txt', 'rb').readlines():
    # print line
    corpus.append(line.strip())

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()

# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

 # 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()

# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()

# 特征向量文本BHTfidf_Result.txt
print('Features length: ' + str(len(word)))
resName = "BHTfidf_Result.txt"
result = codecs.open(resName, 'w', 'utf-8')
for j in range(len(word)):
    result.write(word[j] + ' ')
result.write('\r\n\r\n')

# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weight)):
    # print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
    for j in range(len(word)):
        # print weight[i][j],
        result.write(str(weight[i][j]) + ' ')
    result.write('\r\n\r\n')

result.close()

##########聚类#############
print('Start Kmeans:')
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=3)  # 创建分类器对象
s = clf.fit(weight)  # 用训练数据（权重矩阵）拟合分类器模型
print(s)

# 3个中心点
print(clf.cluster_centers_)

# 每个样本所属的簇
print(clf.labels_)
i = 1
while i <= len(clf.labels_):
    print(i, clf.labels_[i - 1])
    i = i + 1

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
print(clf.inertia_)