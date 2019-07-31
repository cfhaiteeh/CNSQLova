# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import jieba
import cn2an
def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        print("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        print("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data

def load_dataset( mode='train'):
    print("Loading dataset")
    dev_sql, dev_table = load_data('../data/val/val.json', '../data/val/val.tables.json')
    if mode == 'train':
        train_sql, train_table = load_data('../data/train/train.json', '../data/train/train.tables.json')
        return train_sql
    elif mode == 'test':
        test_sql, test_table = load_data('../data/test/test.json', '../data/test/test.tables.json')
        return test_sql
    if mode=='val':
        return dev_sql


def is_number(s):
  try:
    float(s) # for int, long and float
  except ValueError:
    try:
      complex(s) # for complex
    except ValueError:
      return False
  return True
ca=['零','一','二','三','四','五','六','七','八','九','十']

cb=['点','百']
cc=['倍','亿']
noc=['下','共','中','版','甲','级','些','等','胡','搬']
#noc=['下','共','中','版','甲','级','些','等','胡','般','算','股','查','笔','平','手','类','星','大','川','方','时','车','致','州','家','线','生','世','季','期','里','行','益','方','中','般','职','度','文','亚','工','秦','不','集','人','江','批','农','小','盒','位','浮','售','价','景','医','扇','珠','里','论','招','角','合','史','醇','致','墩','平','大','附','石','轮','相','笑','部','口','一','路','冰','峡','道','球','辈','联','店','回','站','杠','经','说','套','锅','王','门','款','冶','加','分','研','德','定','栏','体','院','体','洋','团','宝']

def DataAugment(mode,fileout):

    use_small = False
    sql=load_dataset(mode)

    f = open(fileout, 'w')
    maxc = 0


    for i in range(len(sql)):
        chl = []
        chr = []
        chv = []
        Q = u''.join(sql[i]['question'])
        e = 0
        # cond = test_sql[i]['sql']['conds']
        flag = 1
        if mode=='train':
            cond = sql[i]['sql']['conds']


            for j in range(len(cond)):
                v = u''.join(cond[j][2])
                if v not in Q:
                    flag = 0

        if mode!='train' or flag==0:
            for il in range(len(Q)):
                if il != e:
                    continue
                token = Q[il]
                flag = 0
                e = il + 1
                if token in ca:
                    if il < len(Q) - 1 and Q[il + 1] in noc:
                        continue

                    l = il
                    r = il
                    for ir in range(l, len(Q)):
                        if Q[ir] in ca or Q[ir] in cb:
                            r = ir
                        else:
                            break
                    # print(Q)
                    # print(Q[l:r+1])

                    e = r + 1
                    try:
                        #  print(cn2an.cn2an(Q[l:r+1],'strict'))
                        chv.append(cn2an.cn2an(Q[l:r + 1], 'strict'))
                        chl.append(l)
                        chr.append(r + 1)
                    except:
                        continue
            rQ = Q
            for _i in range(len(chl) - 1, -1, -1):
                Q = Q[0:chl[_i]] + str(chv[_i]) + Q[chr[_i]:len(Q)]
            if (rQ != Q):
                sql[i]['question'] = Q
                maxc += 1
        f.writelines(json.dumps(sql[i], ensure_ascii=False) + '\n')
        # print(dev_sql[i],file=f)
    f.close()
    print(maxc)



if __name__=='__main__':

    DataAugment('val','../data/val/val.json')
    DataAugment('test', '../data/test/test.json')
    DataAugment('train', '../data/train/train.json')


    #print is_number('10.1')

   # tt= translator.translate('我爱你',dest='en',src='zh-CN')
   # print tt.text







