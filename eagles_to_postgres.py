#encoding=utf8
import psycopg2
from elasticsearch import Elasticsearch

#存入数据:先新建表，再插入字典中的数据
conn = psycopg2.connect(database="postgres", user="postgres", password="123456", host="127.0.0.1", port="5432")
print "Opened database successfully"
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS BID
           (ID SERIAL PRIMARY KEY     NOT NULL,
           BID_TYPE         TEXT    NOT NULL ,
           PUBLISH_TIME     TEXT     NOT NULL,
           AREA     TEXT     NOT NULL,
           BUYER     TEXT     NOT NULL,
           TITLE     TEXT     NOT NULL,
           CONTENT     TEXT     NOT NULL);''')
print "Table created successfully"

es = Elasticsearch(hosts=["192.168.50.16:17100"])
#从第0页开始查，查询5条记录
res = es.search(index="pyspider", body={"from":0,
                                        "size":2265,
                                        "query":{"match_all":{}}})

for hit in res['hits']['hits']:
    cur.execute("""INSERT INTO BID
                    (BID_TYPE, PUBLISH_TIME, AREA, BUYER, TITLE, CONTENT)
                    VALUES(%s, %s, %s, %s, %s, %s);""",
                (hit["_source"]['result']['bid_type'],
                 hit["_source"]['result']['publish_time'],
                 hit["_source"]['result']['area'],
                 hit["_source"]['result']['buyer'],
                 hit["_source"]['result']['title'],
                 hit["_source"]['result']['content'],
                 ), )
print "Table insert successfully"
conn.commit()
conn.close()
#hit["_source"]['result']['title']


