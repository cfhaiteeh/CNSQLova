import json
from sqlnet.dbengine import DBEngine
import numpy as np
from tqdm import tqdm

import sqlite3


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    vis_seq = []
    sel_num_seq = []

    gt_type = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]

        sel_num = len(sql['sql']['sel'])
        sel_num_seq.append(sel_num)
        conds_num = len(sql['sql']['conds'])
        q_seq.append([char for char in sql['question']])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))

        # cfhaiteeh
        gt_type.append(table_data[sql['table_id']]['types'])

        ans_seq.append(
            (
                len(sql['sql']['agg']),
                sql['sql']['sel'],
                sql['sql']['agg'],
                conds_num,
                tuple(x[0] for x in sql['sql']['conds']),
                tuple(x[1] for x in sql['sql']['conds']),
                sql['sql']['cond_conn_op'],
            ))
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, gt_type, vis_seq
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, gt_type


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def predict_test(batch_size, table_data, output_path,cs):
    with open('../submit/results_test.jsonl') as inf:
        sqlOutdata = []
        for idx, line in enumerate(inf):
            _sql = json.loads(line.strip())
            _sql['sql'] = _sql['query']
            _sql['question'] = _sql['nlu']
            # print(_sql)
            if ('error') in _sql:
                sqlOutdata.append(sqlOutdata[idx - 1])
            else:
                sqlOutdata.append(_sql)
    
    perm = list(range(len(sqlOutdata)))
    fw = open(output_path, 'w')
    for st in tqdm(range(len(sqlOutdata) // batch_size + 1)):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, gt_type, raw_data = \
            to_batch_seq(sqlOutdata, table_data, perm, st, ed, ret_vis_data=True)

        query_gt, table_ids = to_batch_query(sqlOutdata, perm, st, ed)
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data]  # original question
        # try:

        pred_queriesc, allfsc = genByout(sqlOutdata[st:ed], table_ids, raw_q_seq, gt_type, 'test',cs)
        # print(sql_preds)

        for sql_pred in pred_queriesc:
            fw.writelines(json.dumps(sql_pred, ensure_ascii=False) + '\n')
    fw.close()


def epoch_acc(batch_size, table_data, sql_data, db_path, cs,out=False):
    with open('../submit/results_val.jsonl') as inf:
        sqlOutdata = []
        for idx, line in enumerate(inf):
            _sql = json.loads(line.strip())
            _sql['sql'] = _sql['query']
            _sql['question'] = _sql['nlu']
            # print(_sql)
            if ('error') in _sql:
                sqlOutdata.append(sqlOutdata[idx - 1])
            else:
                sqlOutdata.append(_sql)
    print(len(sql_data))
    print(len(sqlOutdata))
    engine = DBEngine(db_path)
    perm = list(range(len(sqlOutdata)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(len(sql_data) // batch_size + 1)):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, gt_type, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data]  # original question
        # try:

        pred_queriesc, allfsc = genByout(sqlOutdata[st:ed], table_ids, raw_q_seq, gt_type, 'val',cs)

        one_err, tot_err = check_acc(raw_data, pred_queriesc, query_gt, allfsc)

        # except:
        #     badcase += 1
        #     print 'badcase', badcase
        #     continue
        one_acc_num += (ed - st - one_err)
        tot_acc_num += (ed - st - tot_err)

        # Execution Accuracy
        for sql_gt, sql_pred, tid in zip(query_gt, pred_queriesc, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
                                          sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


import synonyms


def genByout(sql_data, table_ids, qu, gt_type, mode,cs):
    def chageCond(cond, tableId, conn, type, raw_Q, vis):

        c = conn.cursor()
        # t = [0, 2,
        #      u'\u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170']
        ti = tableId
        cursor = c.execute("SELECT col_" + str(cond[0] + 1) + "  from Table_" + ti)

        sim = 0
        nword = 'none'

        sa = 0
        lsa = 0
        _sa = 'none'
        _inlength = 0
        lensa = 0
        _instr = ''
        rawstr_1 = cond[2]
        rawstr_2 = cond[2]
        rawstr_3 = cond[2]
        NQ_1 = raw_Q
        NQ_2 = raw_Q
        NQ_3 = raw_Q

        rawstr = cond[2]

        for row in cursor:

            # if str(row[0]) in raw_Q:
            #     sa=1
            #     if len(str(row[0]))>lsa:
            #         lsa=len(str(row[0]))
            #         _sa=row[0]
            if str(row[0]) in raw_Q and (str(row[0]) in vis) == False:

                if (len(str(row[0])) > lensa):
                    lensa = len(str(row[0]))
                    _sa = row[0]
                    lsa = 1.0
                    NQ_2 = raw_Q.replace(str(row[0]), '', 1)
                    rawstr_2 = str(row[0])
                    # break
                continue
            if lensa != 0:
                continue
            if type == 'real' and row[0] != 'None' and len(str(row[0])) != 0 and row[0] != '-' and (
                    str(row[0]) in vis) == False and int(float(row[0])) == float(row[0]):
                if str(int(float(row[0]))) in raw_Q:
                    if len(str(int(float(row[0])))) > _inlength:
                        _inlength = len(str(int(float(row[0]))))
                        _instr = row[0]
                        rawstr_1 = str(row[0])
                        NQ_1 = raw_Q.replace(str(int(float(row[0]))), '', 1)

            _cm = 0.0
            cq = raw_Q
            for everyChar in str(row[0]):
                if everyChar in cq:
                    cq = cq.replace(everyChar, '', 1)
                    _cm += 1
            if _cm / len(str(row[0])) > lsa and (str(row[0]) in vis) == False:
                lsa = _cm / len(str(row[0]))
                _sa = row[0]
                rawstr_2 = str(row[0])
                NQ_2 = cq
            if len(cond[2]) <= 0 or len(str(row[0])) <= 0:
                continue
            _s = synonyms.compare(cond[2], str(row[0]))
            if _s > sim:
                nword = row[0]
                sim = _s
                rawstr_3 = str(row[0])
        # print "Operation done successfully";
        # if sa==1:
        #     cond[2] = str(_sa).decode('utf-8')
        #     return cond
        if _inlength != 0:
            if type == 'real' and int(float(_instr)) == float(_instr):
                _instr = str(int(float(_instr)))
            cond[2] = str(_instr)
            rawstr = rawstr_1
            raw_Q = NQ_1
        elif lsa > 0:
            if type == 'real' and int(float(_sa)) == float(_sa):
                _sa = str(int(float(_sa)))
            cond[2] = str(_sa)
            rawstr = rawstr_2
            raw_Q = NQ_2
        elif sim > 0:

            if type == 'real' and nword != 'None' and int(float(nword)) == float(nword):
                nword = str(int(float(nword)))
                # print nword
            cond[2] = str(nword)
            rawstr = rawstr_3
        return cond, rawstr, raw_Q

    def changeNum(cond, tableId, conn, type):

        c = conn.cursor()
        # t = [0, 2,
        #      u'\u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170 \u5170']
        ti = tableId
        cursor = c.execute("SELECT col_" + str(cond[0] + 1) + "  from Table_" + ti)
        cmpreal = cond[2]
        if cond[2].isdigit() == False:
            return cond
        co = int(cond[2])
        l = 0;
        while (co % 10 == 0 and co != 0):
            co = co / 10
            l += 1
        _maxl = 0
        for row in cursor:
            if type == 'real' and row[0] != 'None' and len(str(row[0])) != 0:
                _co = int(row[0])
                _l = 0
                while (_co % 10 == 0 and _co != 0):
                    _co = _co / 10
                _l += 1
                _maxl = max(_maxl, _l)

        gap = _maxl - l
        # print gap
        if gap >= 3:
            print
            cond[2]
            while (gap > 0):
                cond[2] = cond[2] + '0'
                gap -= 1
            print
            cond[2]
        return cond

    def str_num(cond, raw_Q):
        val = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']
        v = cond[2]
        flag = 0
        for everyChar in v:
            if everyChar in val:
                continue
            else:
                flag = 1
                break
        if flag == 1:
            for l in range(len(raw_Q)):
                if raw_Q[l] in val:
                    r = l
                    for x in range(l, len(raw_Q)):
                        if raw_Q[x] in val:
                            r = x
                        else:
                            break
                    cond[2] = raw_Q[l:r + 1]
        return cond

    def exeDB(DB, tid, sel, agg, conds, cond_conn_op):
        ret_gt = DB.execute(tid, sel, agg, conds, conds)
        # print ret_gt
        return ret_gt

    valIdx = []
    # with open('idx.val','r') as f:
    #     for line in f:
    #         valIdx.append(int(line))
    if mode == 'val':
        conn = sqlite3.connect("../data/val/val.db")
    else:
        conn = sqlite3.connect("../data/test/test.db")
    allfs = []
    B = len(sql_data)
    ret_queries = []
    for b in range(B):

        # print
        cur_query = {}
        # print(sql_data[b])
        cur_query['sel'] = sql_data[b]['sql']['sel']
        cur_query['agg'] = sql_data[b]['sql']['agg']
        sel_num = len(cur_query['sel'])

        cur_query['cond_conn_op'] = sql_data[b]['sql']['cond_conn_op']
        cur_query['conds'] = []
        if (len(sql_data[b]['sql']['conds']) == 0):
            cur_query['cond_conn_op'] = 0
        _s = 0
        used = {}
        for _d in range(25):
            used[_d] = dict()
        raw_Q = ''.join(qu[b])
        okQ = raw_Q
        for idx in range(len(sql_data[b]['sql']['conds'])):
            cur_cond = []

            cur_cond.append(sql_data[b]['sql']['conds'][idx][0])  # where-col

            cur_cond.append(sql_data[b]['sql']['conds'][idx][1])  # where-op
            cur_cond.append(sql_data[b]['sql']['conds'][idx][2])
            types = gt_type[b]

            if cur_cond[1] == 2 and cs=='1':
                cur_cond, rawstr, raw_Q = chageCond(cur_cond, table_ids[b], conn,
                                                    types[sql_data[b]['sql']['conds'][idx][0]], raw_Q,
                                                    used[cur_cond[0]])

                used[cur_cond[0]][rawstr] = 1
            # elif cur_cond[1]!=3:
            #   cur_cond=str_num(cur_cond,okQ)
            #     #cur_cond=changeNum(cur_cond,table_ids[b],conn,types[sql_data[b]['conds'][idx][0]])

            cur_query['conds'].append(cur_cond)
            # _s+=1
            # if _s==cond_num:
            #     break

        ret_queries.append(cur_query)
        # cfhaiteeh
    conn.close()
    return ret_queries, allfs


def check_acc(vis_info, pred_queries, gt_queries, allfs):
    tot_err = sel_num_err = agg_err = sel_err = 0.0
    cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
    for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
        good = True
        sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
        sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

        if where_rela_gt != where_rela_pred:
            good = False
            cond_rela_err += 1

        if len(sel_pred) != len(sel_gt):
            good = False
            sel_num_err += 1

        pred_sel_dict = {k: v for k, v in zip(list(sel_pred), list(agg_pred))}
        gt_sel_dict = {k: v for k, v in zip(sel_gt, agg_gt)}
        if set(sel_pred) != set(sel_gt):
            good = False
            sel_err += 1
        agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
        agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
        if agg_pred != agg_gt:
            good = False
            agg_err += 1

        cond_pred = pred_qry['conds']
        cond_gt = gt_qry['conds']
        if len(cond_pred) != len(cond_gt):
            good = False
            cond_num_err += 1
            cond_col_err += 1
            cond_op_err += 1
            cond_val_err += 1
        else:
            cond_op_pred, cond_op_gt = {}, {}
            cond_val_pred, cond_val_gt = {}, {}
            # update
            colp = []
            colg = []
            for p, g in zip(cond_pred, cond_gt):
                colp.append(p[0])
                colg.append(g[0])

                if (p[0] in cond_op_pred) == False:
                    cond_op_pred[p[0]] = []
                # print('po',p[0])

                cond_op_pred[p[0]].append(p[1])
                if (p[0] in cond_val_pred) == False:
                    cond_val_pred[p[0]] = []
                cond_val_pred[p[0]].append(p[2])

                if (g[0] in cond_op_gt) == False:
                    cond_op_gt[g[0]] = []
                cond_op_gt[g[0]].append(g[1])
                if (g[0] in cond_val_gt) == False:
                    cond_val_gt[g[0]] = []
                cond_val_gt[g[0]].append(g[2])
                # cond_op_pred[p[0]] = p[1]
                # cond_val_pred[p[0]] = p[2]
                # cond_op_gt[g[0]] = g[1]
                # cond_val_gt[g[0]] = g[2]
            colp = sorted(colp)
            colg = sorted(colg)
            if colp != colg:
                cond_col_err += 1
                good = False

            # update
            where_op_pred = [sorted(cond_op_pred[x]) for x in sorted(cond_op_pred.keys())]
            where_op_gt = [sorted(cond_op_gt[x]) for x in sorted(cond_op_gt.keys())]

            if where_op_pred != where_op_gt:
                cond_op_err += 1
                good = False
            # update
            where_val_pred = [sorted(cond_val_pred[x]) for x in sorted(cond_val_pred.keys())]
            where_val_gt = [sorted(cond_val_gt[x]) for x in sorted(cond_val_gt.keys())]
            # if(len(where_val_pred[0])!=1) :
            #     print(where_val_pred)
            cond_pred = sorted(cond_pred, key=lambda l: l[0], reverse=False)
            cond_gt = sorted(cond_gt, key=lambda l: l[0], reverse=False)

            if where_val_pred != where_val_gt:
                cond_val_err += 1
                good = False
                # if set(cond_op_pred.keys()) == set(cond_op_gt.keys()):
                #     if len(where_val_pred)==len(where_val_gt) :
                #         with open("dayuxiaoyuerror.txt","a+") as f:
                #             f.write(u''.join(vis_info[b][0])+'\n')
                #             for j in range(len(where_val_pred)):
                #               if  cond_pred[j][1]==cond_gt[j][1] and cond_gt[j][1]!=2:
                #                 f.write(str(cond_pred[j])+' >>>>><<<<< '+str(cond_gt[j])+'\n')
                #                 f.write( u''.join(where_val_pred[j])+' <<<<<>>>>> '+u''.join(where_val_gt[j])+'\n')
                #             f.write('--------------------------------'+'\n')

        if not good:
            tot_err += 1

    return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err,
                     cond_rela_err)), tot_err

