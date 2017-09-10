# coding=utf8
import warnings
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec



def write_file(segments):
    f = open('result/segment_data.txt', 'w',encoding='utf-8')
    for seg in segments:
        f.write(seg)
        f.write("\n")

    f.close()


if __name__ == '__main__':
    from psql_manage import Psql

    psql = Psql()
    from single_label import get_seg

    ids, segments, labels = get_seg(psql)
    write_file(segments)

    sentences = word2vec.Text8Corpus('result/segment_data.txt')

    model = Word2Vec(sentences, size=200)
    # model.wv[u'食堂']
    # model.wv[u'食品']
    y2 = model.most_similar(u"食堂", topn=20)  # 20个最相关的
    s1 = model.wv.similarity(u'食品', u'食堂')

    print(y2)
    print(s1)
