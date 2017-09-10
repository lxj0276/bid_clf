# coding=utf8
import warnings
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split


def write_file(segments):
    f = open('result/segment_data.txt', 'w', encoding='utf-8')
    for seg in segments:
        f.write(seg)
        f.write("\n")

    f.close()


def get_data():
    from psql_manage import Psql

    psql = Psql()
    from single_label import get_seg

    ids, segments, labels = get_seg(psql)
    return labels, segments


def transform_doc(segments, model):
    doc_vecs = []
    for seg in segments:
        words = seg.split()
        doc_vec = np.array([0.0 for i in range(200)])
        for word in words:
            vec = model.wv[word.strip()]
            doc_vec += vec
        doc_vecs.append(doc_vec)
    print(doc_vecs[1])
    return doc_vecs


def test_word2vec():
    labels, segments = get_data()
    write_file(segments)
    sentences = word2vec.Text8Corpus('result/segment_data.txt')
    model = Word2Vec(sentences, size=200, min_count=1)
    print(type(model.wv[u'食品']))
    print(type(model.wv[u'集贤县']))

    print(model.wv[u'食品'] + model.wv[u'食堂'])
    y2 = model.most_similar(u"食堂", topn=20)  # 20个最相关的
    s1 = model.wv.similarity(u'食品', u'食堂')

    print(y2)
    print(s1)
    doc_vecs = transform_doc(segments, model)
    # return doc_vecs

    X, y = doc_vecs, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from single_label import linear_svc_classifier,benchmark
    clf = linear_svc_classifier()
    # clf = multiclassSVM()
    # clf = nb_classifier()
    benchmark(clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def test_doc2vec():
    ids, segments = get_data()
    model = Doc2Vec(min_count=1, window=15, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(segments.to_array())
    for epoch in range(10):
        model.train(segments.sentences_perm())


if __name__ == '__main__':
    test_word2vec()
