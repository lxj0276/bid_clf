#encoding=utf8
from datetime import datetime
from elasticsearch import Elasticsearch
#连接elasticsearch,默认是9200
es = Elasticsearch(hosts=["192.168.50.16:17100"])

#创建索引，索引的名字是my-index,如果已经存在了，就返回个400，
#这个索引可以现在创建，也可以在后面插入数据的时候再临时创建
es.indices.create(index='my-index',ignore=400)
#{u'acknowledged':True}
#插入数据,(这里省略插入其他两条数据，后面用)index是索引名,doc_type是表名，
es.index(index="my-index",doc_type="test-type",id=01,body={"any":"hello","timestamp":datetime.now()})
es.index(index="my-index",doc_type="test-type",id=02,body={"any":"Should I stick to it?","timestamp":datetime.now()})
#{u'_type':u'test-type',u'created':True,u'_shards':{u'successful':1,u'failed':0,u'total':2},u'_version':1,u'_index':u'my-index',u'_id':u'1}
#也可以，在插入数据的时候再创建索引test-index
#es.index(index="test-index",doc_type="test-type",id=42,body={"any":"should i persist","timestamp":datetime.now()})

#查询数据，两种get and search
#get获取
#res = es.get(index="test-index", doc_type="test-type", id=42)
#print(res)
#{u'_type': u'test-type', u'_source': {u'timestamp': u'2016-01-20T10:53:36.997000', u'any': u'data01'}, u'_index': u'my-index', u'_version': 1, u'found': True, u'_id': u'1'}
#print(res['_source'])
#{u'timestamp': u'2016-01-20T10:53:36.997000', u'any': u'data01'}
#search获取
#res = es.search(index="my-index", body={"query":{"match_all":{}}})
#print(res)