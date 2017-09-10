import psycopg2


class Psql:
    def __init__(self):
        self.conn = psycopg2.connect(database="postgres", user="postgres", password="123456", host="127.0.0.1",
                                     port="5432")
        print("Opened database successfully")
        self.cur = self.conn.cursor()


    def execute_sql(self,sql):
        """
        执行一条查询语句，并返回
        :param sql:
        :return: 返回查询结果集
        """
        print(sql)
        self.cur.execute(sql)
        rows = self.cur.fetchall()
        return rows

    def execute_commit(self,sql):
        """
        执行一条sql语句，并提交
        :param sql:
        :return:
        """
        print(sql)
        try:
            self.cur.execute(sql)
            self.conn.commit
        except Exception as e:
             print(e)
             self.conn.rollback()


    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_titles_data(self):
        sql = 'select id,title from bid'
        rows = self.execute_sql(sql)
        datas = {}
        for row in rows:
            id = row[0]
            title = row[1]
            datas[id] = title
        return datas

    def get_train_data(self):
        self.cur.execute('select id,title,labels from train')
        rows = self.cur.fetchall()
        ids = []
        titles = []
        labels = []
        for row in rows:
            id = row[0]
            title = row[1].strip()
            label = row[2].strip().split(',')
            ids.append(id)
            titles.append(title)
            labels.append(label)
        return ids, titles, labels

    def single_label_date(self):
        """
        获取数据：多标签转换成单标签
        :return:
        """
        self.cur.execute('select id,title,labels from train')
        rows = self.cur.fetchall()
        ids = []
        titles = []
        labels = []
        for row in rows:
            id = row[0]
            title = row[1].strip()
            label = row[2].strip().split(',')[0]
            ids.append(id)
            titles.append(title)
            labels.append(label)
        return ids, titles, labels

    def update_label(self, ids, labels):
        try:
            for i in range(len(ids)):
                id = ids[i]
                label = labels[i]
                sql = "update bid set label = " + str(label) + " where id = " + str(id)
                self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()
        pass


if __name__ == '__main__':
    psql = Psql()
    psql.get_dbdate()

    psql.close()
