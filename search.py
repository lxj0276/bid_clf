#encoding=utf8
from elasticsearch import Elasticsearch
#连接elasticsearch,默认是9200
es = Elasticsearch(hosts=["192.168.50.16:17100"])
#从第0页开始查，查询5条记录
res = es.search(index="pyspider", body={"from":0,
                                        "size":5,
                                        "query":{"match_all":{}}})
#print(res)

for hit in res['hits']['hits']:
    #print(hit["_id"])
    #print(hit["_source"])
    #print(hit["_source"]['result'])
    print(hit["_source"]['result']['title'])
#res = es.search(index="pyspider", body={'query':{'match':{'any':'data'}}}) #获取any=data的所有值
#print(res)