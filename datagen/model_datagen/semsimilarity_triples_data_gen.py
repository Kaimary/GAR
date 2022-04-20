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
from config import DIR_PATH, SERIALIZE_DATA_DIR, SEMSIMILARITY_TRIPLE_DATA_GZ_FILE, REWRITE
from spider_utils.recall_checker_utils import RecallChecker


@click.command()
@click.argument("dataset_name", default="spider")
@click.argument("train_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("dev_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("num_sql", default=100)
@click.argument("overwrite", default=1)
def main(dataset_name, train_file, dev_file, tables_file, db_dir, num_sql, overwrite):
    if dataset_name == "geo":
        output_train_data = generate_triples_from_spider_dataset(
            dataset_name, train_file, tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite,
            split='train',
            per_db_threshold=2000
        )
        output_dev_data = generate_triples_from_spider_dataset(
            dataset_name, dev_file, tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite, split='dev',
            per_db_threshold=100
        )
    else:
        output_train_data = generate_triples_from_spider_dataset(
            dataset_name, train_file, tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite
        )
        output_dev_data = generate_triples_from_spider_dataset(
            dataset_name, dev_file, tables_file, db_dir, sql_generation_num=num_sql, overwrite=overwrite, split='dev',
            per_db_threshold=1
        )
    print(f"Generate semantic similarity training triple data: {len(output_train_data)}.")
    print(f"Generate semantic similarity dev triple data: {len(output_dev_data)}.")
    # Write out into file
    tmp_file = "./tmp.tsv"
    with open(tmp_file, 'wt', ) as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        tsv_writer.writerow(['split', 'genre', 'dataset', 'year', 'sid', 'score', 'sentence1', 'sentence2'])
        tsv_writer.writerows(output_train_data)
        tsv_writer.writerows(output_dev_data)
    print(
        f"Overwrite semantic similarity triple data into file: {SEMSIMILARITY_TRIPLE_DATA_GZ_FILE.format(dataset_name)}.")
    with open(tmp_file, 'rb') as f_in, gzip.open(DIR_PATH + SEMSIMILARITY_TRIPLE_DATA_GZ_FILE.format(dataset_name),
                                                 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(tmp_file)
    print(f"Overwrite semantic similarity triple data succeed!")

    return


# Compare the SQL structures and calculate a similarity score between the two.
def calculate_score(g_sql, p_sql):
    evaluator = Evaluator()
    total_score = 5.0
    # We first check the sources of the two sqls, we assume it dominants the similarity score.
    if len(g_sql['from']['table_units']) > 0:
        label_tables = sorted(g_sql['from']['table_units'])
        pred_tables = sorted(p_sql['from']['table_units'])
        if label_tables != pred_tables:
            total_score -= 1.0
        elif len(g_sql['from']['conds']) > 0:
            label_joins = sorted(g_sql['from']['conds'], key=lambda x: str(x))
            pred_joins = sorted(p_sql['from']['conds'], key=lambda x: str(x))
            if label_joins != pred_joins:
                total_score -= 0.5
    partial_scores = evaluator.eval_partial_match(deepcopy(p_sql), deepcopy(g_sql))
    # Next we use 7 of 10 categories from partial scores to do the comparison: 
    # 1)select 2)where 3)group 4)order 5)and/or 6)IUE 7)keywords
    for category, score in partial_scores.items():
        if score['f1'] != 1:
            if category == "keywords":
                total_score -= 0.5
            elif category == "select":
                total_score -= 1.0
            elif category == "where":
                total_score -= 0.5
            elif category == "group":
                total_score -= 0.5
            elif category == "order":
                total_score -= 0.5
            elif category == "and/or":
                total_score -= 0.2
            elif category == "IUEN":
                total_score -= 0.8

    return total_score


# Generate semantic similarity triple data for spider dataset(Train/Dev/Test)
def generate_triples_from_spider_dataset(
        dataset_name, file_path, tables_file, db_dir, sql_generation_num=10000, overwrite=1, split='train',
        per_db_threshold=40):
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
    all_schema = get_all_schema(file_path=tables_file)
    # Use to check miss rate for each of the phase
    checker = RecallChecker(file_path, tables_file, db_dir, kmaps)
    with open(file_path, "r") as data_file:
        data = json.load(data_file)
        for ex in data:
            num_total_count += 1
            db_id = ex['db_id']
            # if db_id == 'flight_4': flag = True 
            # if not flag: continue
            if db_id == 'company_1' or db_id == 'baseball_1' or \
                    db_id == 'customer_complaints':
                continue
            # ---------------------------------- FOR DEBUG ----------------------------------
            # if db_id != 'world_1' and db_id != 'car_1':
            #     continue
            # if db_id != 'museum_visit' and db_id != 'wta_1' and db_id != 'student_transcripts_tracking' \
            #         and db_id != 'dog_kennels':
            #     continue
            # ********************************** END DEBUG **********************************
            # if db_id == 'product_catalog' or db_id == 'company_1' or db_id == 'coffee_shop' or db_id == 'chinook_1':
            #     continue
            # If current instance is the first instance of one database, 
            # we firstly generate negative triples of previous database and then clear up all variables;

            # 切换了一个新的db中的内容
            is_switching_db_records = pre_db_id != "" and pre_db_id != db_id
            # 切换到新db或者所有记录遍历完
            if is_switching_db_records or num_total_count == len(data):
                # For each db, we randomly select fixed number of sql-dialects as negative instances.
                pairs = list(zip(gold_sqls, nls))
                threshold = per_db_threshold
                per_db_per_range_threshold = per_db_threshold / 5 * 1.2
                triple_0_1_count = 0
                triple_1_2_count = 0
                triple_2_3_count = 0
                triple_3_4_count = 0
                triple_4_5_count = 0
                timeout = False
                time_start = time.time()
                while threshold > 0:
                    if timeout:
                        break
                    gold_sql, nl = random.choice(pairs)
                    idx = nls.index(nl)
                    for irr_sql, irr_dialect in zip(sqls, dialects):
                        p_sql = rebuild_sql(pre_db_id, db_dir, sql_nested_query_tmp_name_convert(irr_sql),
                                            kmaps)
                        g_sql = rebuild_sql(pre_db_id, db_dir, sql_nested_query_tmp_name_convert(gold_sql),
                                            kmaps)
                        score = calculate_score(g_sql, p_sql)
                        if 0 <= score < 1.0 and triple_0_1_count < per_db_per_range_threshold:
                            triple_0_1_count += 1
                        elif 1 <= score < 2.0 and triple_1_2_count < per_db_per_range_threshold:
                            triple_1_2_count += 1
                        elif 2 <= score < 3.0 and triple_2_3_count < per_db_per_range_threshold:
                            triple_2_3_count += 1
                        elif 3 <= score < 4.0 and triple_3_4_count < per_db_per_range_threshold:
                            triple_3_4_count += 1
                        elif 4 <= score < 5.0 and triple_4_5_count < per_db_per_range_threshold:
                            triple_4_5_count += 1
                        else:
                            # We break the generation loop if the time elapse exceeds 15mins.
                            time_end = time.time()
                            elapsed = (time_end - time_start) / 60
                            if elapsed > 5:
                                timeout = True
                                break
                            else:
                                continue
                        threshold -= 1
                        output.append([split, "dialects", "spider", "2021", f"{pre_db_id}{idx + 1}", score, nl,
                                       irr_dialect])
                        break
                print(f"output:{len(output)}, pre_db_id:{pre_db_id}")
                # Clear up
                index = 0
                nls.clear()
                gold_sqls.clear()
                sqls.clear()
                dialects.clear()

            # 进入每一个db时处理
            if pre_db_id == "" or pre_db_id != db_id:
                # Generate synthesis sql-dialects
                pre_db_id = db_id
                print(f"db_id: {pre_db_id}")
                db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
                if not os.path.isfile(db_file):
                    schema = get_schema_from_json(db_id, tables_file)
                else:
                    schema = get_schema(db_file)
                _, table, table_dict = read_single_dataset_schema(tables_file, pre_db_id)
                db_data_path = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}/{pre_db_id}_{sql_generation_num}.txt'
                if not os.path.exists(db_data_path) or bool(overwrite):
                    # sql_map_set = set()
                    db_schema = DBSchema(pre_db_id, all_schema, db_dir)
                    g = Generator(dataset_name, pre_db_id, db_schema, file_path, tables_file, db_dir, kmaps,
                                  stage=split)

                    # def generator():
                    # while len(sqls) < sql_generation_num:
                    #     yield

                    # for _ in tqdm(generator()): 
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

                    if split == 'dev':
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
                else:
                    all_lines = []
                    datafile = open(db_data_path, 'r')
                    for line in datafile.readlines():
                        sql, dialect = line.split('\t')
                        sqls.append(sql.strip())

                        if REWRITE == True:
                            sql1 = sql_nested_query_tmp_name_convert(sql)
                            _, sql_dict, schema_ = disambiguate_items2(tokenize(sql1), schema, table,
                                                                       allow_aliases=False)
                            dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                            line = sql + '\t' + dialect + '\n'
                            all_lines.append(line)

                        dialects.append(dialect.strip())
                    datafile.close()

                    if REWRITE == True:
                        datafile = open(db_data_path, 'w')
                        datafile.writelines(all_lines)
                        datafile.close()

                    if split == 'dev':
                        miss_rate = checker.check_sqlgen_recall(pre_db_id, sqls)
                        print(f"db: {db_id} miss rate:{miss_rate}@{sql_generation_num}")

            # For each instance, we add to the file as the positive triple.
            nl = ex['question']
            nls.append(nl)
            ex = fix_number_value(ex)
            # Reconstruct gold sql by masking out value tokens
            gold_sql = fix_query_toks_no_value(ex['query_toks_no_value'])
            gold_sql = sql_nested_query_tmp_name_convert(gold_sql)
            gold_sqls.append(gold_sql)
            _, gold_sql_dict, schema_ = disambiguate_items2(tokenize(gold_sql), schema, table, allow_aliases=False)
            gold_dialect = convert_sql_to_dialect(gold_sql_dict, table_dict, schema_)
            # Add nl-gold dialect as a row
            output.append([split, "dialects", "spider", "2021", f"{db_id}000{index}", 5.000, nl, gold_dialect])
            index += 1

    checker.print_sqlgen_total_result(num_total_count, sql_generation_num)
    checker.export_sqlgen_miss_sqls()
    return output


if __name__ == "__main__":
    main()
