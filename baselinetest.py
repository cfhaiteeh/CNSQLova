import torch
from baselineutils import *

import argparse
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

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data
def load_dataset(toy=False, use_small=False, mode='train'):
    print("Loading dataset")
    val_sql, val_table = load_data('../data/val/val.json', '../data/val/val.tables.json', use_small=use_small)
    val_db = '../data/val/val.db'

    test_sql, test_table = load_data('../data/test/test.json', '../data/test/test.tables.json', use_small=use_small)
    test_db = '../data/test/test.db'
    print('over')
    return val_sql, val_table, val_db, test_sql, test_table, test_db
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
    parser.add_argument('--toy', action='store_true', help='Small batchsize for fast debugging.')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention.')
    parser.add_argument('--train_emb', action='store_true', help='Use trained word embedding for SQLNet.')
    parser.add_argument('--output_dir', type=str, default='', help='Output path of prediction result')
    parser.add_argument('--mode', type=str, default='', help='Output path of prediction result')
    parser.add_argument('--cs', type=str, default='', help='Output path of prediction result')
    args = parser.parse_args()

    n_word=300
    if args.toy:
        use_small=True
        gpu=args.gpu
        batch_size=16
    else:
        use_small=False
        gpu=args.gpu
        batch_size=64

    val_sql, val_table, val_db, test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')
    print(args.mode)
    if args.mode=='val':
        dev_acc = epoch_acc(batch_size, val_table, val_sql, val_db, args.cs,out=True)
        print(
            'Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f' % (
                dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6],
                dev_acc[0][7]))
        print('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))

    if args.mode == 'test':
        print("Start to predict test set")
        predict_test( batch_size, test_table, args.output_dir,args.cs)
        print("Output path of prediction result is %s" % args.output_dir)
