import os
import json
import csv
import random
import click
import time
import gzip
from copy import deepcopy

from spider_utils.utils import fix_number_value, read_single_dataset_schema, fix_query_toks_no_value, \
    disambiguate_items2, DBSchema, get_all_schema
from datagen.sqlgen.sqlgen import Generator
from datagen.sqlgen.utils.sql_tmp_update import sql_nested_query_tmp_name_convert, use_alias
from datagen.dialectgen.bst_traverse import convert_sql_to_dialect
from spider_utils.evaluation.process_sql import get_schema, get_schema_from_json, tokenize
from spider_utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql
from config import DIR_PATH, SERIALIZE_DATA_DIR, SEMSIMILARITY_TRIPLE_DATA_FINETUNE_GZ_FILE, REWRITE
from spider_utils.recall_checker_utils import RecallChecker



def main():
    dataset_name = 'geo'
    tables_file = 'sqlgenv2/datasets/geo/tables.json'
    db_dir = 'sqlgenv2/datasets/geo/database'
    num_sql = 20000
    overwrite = 1
    generate_serialization_from_dataset(
        dataset_name, 'sqlgenv2/datasets/geo/train.json', tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite, split='train'
    )
    generate_serialization_from_dataset(
        dataset_name, 'sqlgenv2/datasets/geo/dev.json', tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite, split='dev'
    )
    generate_serialization_from_dataset(
        dataset_name, 'sqlgenv2/datasets/geo/test.json', tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite, split='test'
    )

    return


# Generate semantic similarity triple data for spider dataset(Train/Dev/Test)
def generate_serialization_from_dataset(
        dataset_name, file_path, tables_file, db_dir, sql_generation_num=20000, overwrite=1, split='train'):
    gold_sqls = []
    nls = []
    sqls = []
    dialects = []
    pre_db_id = ""
    num_total_count = 0
    index = 0
    output = []
    # flag = False
    kmaps = build_foreign_key_map_from_json(tables_file)
    all_schema = get_all_schema(tables_file)
    # Use to check miss rate for each of the phase
    checker = RecallChecker(file_path, tables_file, db_dir, kmaps)
    with open(file_path, "r") as data_file:
        data = json.load(data_file)
        for ex in data:
            num_total_count += 1
            db_id = ex['db_id']   
            
            # 进入每一个db时处理
            if pre_db_id == "" or pre_db_id != db_id:
                # Generate synthesis sql-dialects
                pre_db_id = db_id
                db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
                # if not os.path.isfile(db_file):
                schema = get_schema_from_json(db_id, tables_file)
                # else:
                #     schema = get_schema(db_file)
                _, table, table_dict = read_single_dataset_schema(tables_file, pre_db_id)
                db_data_path = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}/{pre_db_id}_{sql_generation_num}.txt'
                if not os.path.exists(db_data_path) or bool(overwrite):
                    # sql_map_set = set()
                    db_schema = DBSchema(pre_db_id, all_schema, db_dir)                    
                    g = Generator(dataset_name, pre_db_id, db_schema, file_path, tables_file, db_dir, kmaps,
                                  stage=split)                                        
                    sqls = g.generate_sql(sql_generation_num)
                    # sql1 = sql_nested_query_tmp_name_convert(sql)
                    # sql_map = json.dumps(rebuild_sql(db_id, db_dir, sql1, kmaps))
                    # if sql_map in sql_map_set:
                    #     continue
                    # else:
                    # sql_map_set.add(sql_map)
                    # sqls.append(sql)
                    for sql in sqls:
                        sql1 = sql_nested_query_tmp_name_convert(sql)
                        sql1 = use_alias(sql1)
                        _, sql_dict, schema_ = disambiguate_items2(tokenize(sql1), schema, table, allow_aliases=False)
                        dialects.append(convert_sql_to_dialect(sql_dict, table_dict, schema_))

                    
                    miss_rate = checker.check_sqlgen_recall(pre_db_id, sqls)
                    print(f"db: {db_id} miss rate:{miss_rate}@{sql_generation_num}")

                    datafile = open(db_data_path, 'w')
                    for sql, dialect in zip(sqls, dialects):
                        # write line to output file
                        datafile.write(sql)
                        datafile.write('\t')
                        datafile.write(dialect)
                        datafile.write("\n")
                    datafile.close()
                # Read from the existing file

            

    checker.print_sqlgen_total_result(num_total_count, sql_generation_num)
    checker.export_sqlgen_miss_sqls()
    return


if __name__ == "__main__":
    main()
