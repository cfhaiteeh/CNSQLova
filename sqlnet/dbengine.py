# From original SQLNet code.
# Wonseok modified. 20180607

import records
import re
from babel.numbers import parse_decimal, NumberFormatError


schema_re = re.compile(r'\((.+)\)') # group (.......) dfdf (.... )group
num_re = re.compile(r'[-+]?\d*\.\d+|\d+') # ? zero or one time appear of preceding character, * zero or several time appear of preceding character.
# Catch something like -34.34, .4543,
# | is 'or'

agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
cond_ops = ['>', '<', '=', '!=']
rela_dict = [' ',' AND ', ' OR ']
class DBEngine:

    def __init__(self, fdb):
        #fdb = 'data/test.db'
        self.db = records.Database('sqlite:///{}'.format(fdb))

  
    def execute(self, table_id, select_index, aggregation_index, conditions,condition_relation, lower=True):
        # if not table_id.startswith('table'):
        #     table_id = 'table_{}'.format(table_id.replace('-', '_'))
        if not table_id.startswith('Table'):
            table_id = 'Table_{}'.format(table_id.replace('-', '_'))
        #table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n','')
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        # for tup in schema_str.split(', '):
        #     c, t = tup.split()
        #     schema[c] = t
        for tup in schema_str.split(','):
            c, t = tup.split(' ')
            schema[c] = t


        # select = 'col_{}'.format(select_index+1)
        # agg = agg_ops[aggregation_index]
        # if agg:
        #     select = '{}({})'.format(agg, select)

        select_part = ""
        for sel, agg in zip(select_index, aggregation_index):
            select_str = 'col_{}'.format(sel + 1)
            agg_str = agg_ops[agg]
            if agg:
                select_part += '{}({}),'.format(agg_str, select_str)
            else:
                select_part += '({}),'.format(select_str)
        select_part = select_part[:-1]

        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            #if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):

            if schema['col_{}'.format(col_index + 1)] == 'real' and not isinstance(val, (int, float)):

                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)

                except NumberFormatError as e:
                    try:
                        val = float(num_re.findall(val)[0]) # need to understand and debug this part.
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass
            where_clause.append('col_{} {} :col_{}'.format(col_index+1, cond_ops[op], col_index+1))
            where_map['col_{}'.format(col_index+1)] = val
        where_str = ''
        if len(where_clause)>1 and condition_relation==0:
            condition_relation=1
        if where_clause:
            where_str = 'WHERE ' + rela_dict[condition_relation].join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select_part, table_id, where_str)
        #print query
        out = self.db.query(query, **where_map)


        return [o.result for o in out]
    