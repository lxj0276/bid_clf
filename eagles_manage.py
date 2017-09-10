from elasticsearch import Elasticsearch

def get_data():
    es = Elasticsearch(hosts=["192.168.50.16:17100"])
    #从第0页开始查，查询5条记录
    res = es.search(index="pyspider", body={"from":0,
                                            "size":7265,
                                            "query":{"match_all":{}}})
    titles = []
    for hit in res['hits']['hits']:
        titles.append(hit["_source"]['result']['title'])
    return titles