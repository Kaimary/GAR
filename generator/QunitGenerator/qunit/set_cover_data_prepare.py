# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:10
# @Author  : 
# @Email   : 
# @File    : set_cover_data_prepare.py
# @Software: PyCharm
import click
import os
from configs.config import SERIALIZE_DATA_DIR, QUNITS_SET_COVER_MINUS_FILE
from .unit_extract import *


def get_generated_sql(file_path):
    """
    load generated sqls
    @param file_path: sql file path
    @return: generated sqls in a list
    """
    with open(file_path) as f:
        sqls = [line.split('\t')[0] for line in f.readlines()]
    return sqls


@click.command()
@click.argument("sql_num", type=int)
def set_cover_data(sql_num):
    db_schemas = get_all_schema()
    spider_set_cover = dict()
    count = 0
    for key in db_schemas.keys():
        count += 1
        print(f'[INFO] Handling DB ({count:>3}/{len(db_schemas)}): {key}\r', end='')
        generated_sql_file = DIR_PATH + SERIALIZE_DATA_DIR.format(dataset_name) + os.sep + f'{key}_{sql_num}.txt'
        # no exists generated sqls
        if not os.path.exists(generated_sql_file):
            print(f'[INFO] No generated sql file {generated_sql_file.split(os.sep)[-1]}')
            continue
        print(f'[INFO] Handling DB ({count:>3}/{len(db_schemas)}): {key}')
        db_schema = DBSchema(key, db_schemas)

        spider_pattern = SpiderPattern(get_all_sql_by_id(key), db_schema)
        spider_set_cover[key] = spider_pattern.sql_unit_dict

        spider_pattern_generated = SpiderPattern(get_generated_sql(generated_sql_file), db_schema)
        spider_set_cover[key + '_generated'] = spider_pattern_generated.sql_unit_dict

    with open(DIR_PATH + QUNITS_SET_COVER_MINUS_FILE, 'w') as f:
        json.dump(spider_set_cover, f, indent=4)


if __name__ == '__main__':
    set_cover_data()
