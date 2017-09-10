#encoding=utf8
import jieba
import codecs#这是啥
#from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from psql_manage import Psql

#######################
def load_stopwords():
    input = open('chinese_stopword.txt', 'r',encoding= 'utf-8')
    lines = []
    for line in input.readlines():
        lines.append(line.strip())
    return lines


########################


stop_words = load_stopwords()
psql =Psql()
datas = psql.get_titles_data()
ids = []
titles = []
for key in datas.keys():
    ids.append(key)
    titles.append(datas[key])
corpus = []
for title in titles:
    seg_list = []
    for word in jieba.cut(title):
        if word in stop_words:
            # print(word)
            continue
        seg_list.append(word)
    #print " /".join(seg_list)
    corpus.append(' '.join(seg_list))

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()#词向量转换
# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()#统计词权重tfidf值
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
#f.__next__() == f.next()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))#得到tfidf矩阵
 # 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()



# 特征向量文本BHTfidf_Result.txt
print('Features length: ' + str(len(word)))
resName = "BH_Tfidf_Result.txt"
result = codecs.open(resName, 'w', 'utf-8')
#写入特征word
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

clf = KMeans(n_clusters=7)  # 创建分类器对象
s = clf.fit(weight)  # 用数据（权重矩阵）拟合分类器模型
print(s)

# 3个中心点
print(clf.cluster_centers_)

# 每个样本所属的簇
#print(clf.labels_)
i = 1
while i <= len(clf.labels_):
    print(ids[i-1],titles[i-1], clf.labels_[i - 1])
    i = i + 1

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
print(clf.inertia_)

psql.update_label(ids,clf.labels_)