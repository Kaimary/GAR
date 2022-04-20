import sys

import datetime
import numpy as np
import json
import os
import click
import random
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from spider_utils.utils import fix_number_value, read_single_dataset_schema, fix_query_toks_no_value, \
    disambiguate_items2, \
    DBSchema, get_all_schema
from datagen.sqlgen.sqlgen import Generator
from datagen.sqlgen.utils.sql_tmp_update import sql_nested_query_tmp_name_convert
from datagen.dialectgen.bst_traverse import convert_sql_to_dialect
from spider_utils.evaluation.process_sql import get_schema, get_schema_from_json, tokenize
from spider_utils.evaluation.evaluate import build_foreign_key_map_from_json, rebuild_sql
from config import DIR_PATH, SERIALIZE_DATA_DIR, SENTENCE_EMBEDDER_MODEL_DIR, \
    SENTENCE_EMBEDDER_MODEL_DIMENSION, \
    TEST_DATA_FILE_NAME, CANDIDATE_MISS_FILE_NAME, SQL_MISS_FILE_NAME, REWRITE
from spider_utils.recall_checker_utils import RecallChecker


@click.command()
@click.argument("dataset_name", default="spider")
@click.argument("dev_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("num_sql", default=100)
@click.argument("candidate_num", default=300)
@click.argument("overwrite", default=0)
@click.argument("experiment_dir_name", default="/output/spider/reranker/default_experiment_dir")
@click.argument("candidate_embedding_model_name", default="")
@click.argument("is_82", type=click.BOOL, default=False)
def main(dataset_name, dev_file, tables_file, db_dir, num_sql, candidate_num, overwrite,
         experiment_dir_name, candidate_embedding_model_name, is_82):
    embedder = SentenceTransformer(
        DIR_PATH + SENTENCE_EMBEDDER_MODEL_DIR.format(dataset_name) + '_' + candidate_embedding_model_name)
    kmaps = build_foreign_key_map_from_json(tables_file)
    os.makedirs(experiment_dir_name)

    # detect if 82 setup
    stage = "dev"
    if is_82:
        print("[INFO] 82 environment set detected")
        stage = "dev_80"
    output_test_data, miss_candidate_log = generate_rerank_data_from_spider_dataset(embedder, kmaps, dataset_name, dev_file, tables_file, db_dir,
                                                                experiment_dir_name, dev=True, max_instances=None,
                                                                sql_generation_num=num_sql,
                                                                candidate_num=candidate_num, overwrite=overwrite,
                                                                stage=stage)
    #TODO:
    with open(experiment_dir_name + '/' + TEST_DATA_FILE_NAME, 'w') as outfile:
        json.dump(output_test_data, outfile, indent=4)
    with open(experiment_dir_name + '/miss_candidate_log.json', 'w') as outfile:
        json.dump(miss_candidate_log, outfile, indent=4)

    return

    
def generate_rerank_data_from_spider_dataset(
        embedder, kmaps, dataset_name, dataset_file, tables_file, db_dir, experiment_dir_name, dev=False, 
        max_instances=None,
        sql_generation_num=10000, candidate_num=200, overwrite=0,
        stage="dev"
):
    output = []
    d = int(SENTENCE_EMBEDDER_MODEL_DIMENSION)
    num_total_count = 0
    db_count = 0
    current_db_id = ""
    sqls = []
    dialects = []
    dev_count = 0
    all_schema = get_all_schema(tables_file)
    checker = RecallChecker(dataset_file, tables_file, db_dir, kmaps)

    miss_candidates_log = []

    with open(dataset_file, "r") as data_file:
        data = json.load(data_file)
        for ex in data:
            num_total_count += 1
            db_id = ex['db_id']
            # if db_id != "twitter_1" and db_id != "concert_singer" and not flag: continue
            # flag = True
            if db_id == 'company_1' or db_id == 'baseball_1' or db_id == 'customer_complaints': continue
            ex = fix_number_value(ex)
            original_gold_sql = ex['query']
            gold_sql = fix_query_toks_no_value(ex['query_toks_no_value'])

            # get gold dialect
            db_file = os.path.join(db_dir, db_id, db_id + ".sqlite")
            if not os.path.isfile(db_file):
                schema = get_schema_from_json(db_id, tables_file)
            else:
                schema = get_schema(db_file)
            _, table, table_dict = read_single_dataset_schema(tables_file, db_id)
            gold_sql1 = sql_nested_query_tmp_name_convert(gold_sql)
            _, sql_dict, schema_ = disambiguate_items2(tokenize(gold_sql1), schema, table, allow_aliases=False)
            gold_dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)

            question = ex['question']
            if not current_db_id or current_db_id != db_id:
                if current_db_id:
                    checker.print_candidategen_result(current_db_id, candidate_num)

                current_db_id = db_id
                print(f"db_id: {current_db_id}")
                db_count += 1
                print(f"db_count: {db_count}")
                sqls.clear()
                dialects.clear()

                db_data_path = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}/{current_db_id}_{sql_generation_num}.txt'
                if not os.path.exists(db_data_path) or bool(overwrite):
                    db_schema = DBSchema(current_db_id, all_schema, db_path=db_dir)
                    if stage.endswith('_80'):
                        dataset_file = dataset_file.replace('_20.json', '_80.json')
                    g = Generator(dataset_name, current_db_id, db_schema, dataset_file, tables_file, db_dir, kmaps,
                                  stage=stage)

                    # def generator():
                    #     while len(sqls) < sql_generation_num:
                    #         yield

                    # for _ in tqdm(generator()): 
                    sqls = g.generate_sql(sql_generation_num)
                    sqls = list(sqls)
                    # sql1 = sql_nested_query_tmp_name_convert(sql)
                    # sql_map = json.dumps(rebuild_sql(db_id, db_dir, sql1, kmaps))
                    # if sql_map in sql_map_set:
                    #     continue
                    # else:
                    for sql in sqls:
                        # sql_map_set.add(sql_map)
                        # sqls.append(sql)
                        sql1 = sql_nested_query_tmp_name_convert(sql)
                        _, sql_dict, schema_ = disambiguate_items2(tokenize(sql1), schema, table, allow_aliases=False)
                        dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                        dialects.append(dialect)
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

                sqlgen_miss = checker.check_sqlgen_recall(db_id, sqls)

                # start_time = datetime.datetime.now()
                dialect_embeddings = embedder.encode(dialects)
                # end_time = datetime.datetime.now()
                # print(f"Embeddding dialects complete: {end_time-start_time}")

                # start_time = datetime.datetime.now()
                index = faiss.IndexFlatL2(d)
                # print(index.is_trained)
                index.add(np.stack(dialect_embeddings, axis=0))
                # end_time = datetime.datetime.now()
                # print(f"Indexing added: {end_time-start_time}")
                # print(index.ntotal)

            question_embedding = embedder.encode(question)
            # D, I = index.search(np.stack(query_embeddings, axis=0), k)     # actual search
            # print(I)                   # neighbors of the 5 first queries
            start_time = datetime.datetime.now()
            distances, indices = index.search(np.asarray(question_embedding).reshape(1, d), candidate_num + 1)
            candidate_dialects = [dialects[indices[0, idx]] for idx in range(0, candidate_num)]
            end_time = datetime.datetime.now() 
            # print(f"Candidate dialects searching complete: {end_time-start_time}")
            candidate_sqls = [sqls[indices[0, idx]] for idx in range(0, candidate_num)]

            gold_sql_index = checker.check_add_candidategen_miss(db_id, candidate_sqls, gold_sql, original_gold_sql,
                                                                 gold_dialect, question, db_dir, kmaps)
            # Swap to make sure that the gold one is the last element.
            if gold_sql_index != -1:
                candidate_sqls[gold_sql_index], candidate_sqls[candidate_num - 1] = candidate_sqls[candidate_num - 1], \
                                                                                    candidate_sqls[gold_sql_index]
                candidate_dialects[gold_sql_index], candidate_dialects[candidate_num - 1] = candidate_dialects[
                                                                                                candidate_num - 1], \
                                                                                            candidate_dialects[
                                                                                                gold_sql_index]

            if dev:
                dev_count += 1
                candidates = candidate_dialects
                if gold_sql_index != -1:
                    labels = [0] * (candidate_num - 1) + [1]
                else:
                    labels = [0] * (candidate_num - 1) + [0]
                    miss_candidates_log.append({
                        'num': num_total_count,
                        'question': question,
                        'gold_dialect': gold_dialect,
                        'gold_sql': gold_sql,
                        'top 10 predict dialect': [dia for dia in candidates[:10]],
                        'top 10 predict sql': [sql for sql in candidate_sqls[:10]]
                    })
                # Shuffle the list
                # c = list(zip(candidates, labels, candidate_sqls))
                # random.shuffle(c)
                # candidates, labels, candidate_sqls = zip(*c)
                ins = {
                    "db_id": db_id,
                    "question": question,
                    "candidates": candidates,
                    "candidate_sqls": candidate_sqls,
                    "labels": labels
                }
                output.append(ins)
            else:
                for j in range(0, (int)(candidate_num / 10)):
                    coin = random.choices([1, 0], weights=(30, 70))[0]
                    if bool(coin):
                        candidates = candidate_dialects[j * 10:(j + 1) * 10] + [candidate_dialects[-1]]
                        labels = [0] * 10 + [1]
                        # Shuffle the list
                        c = list(zip(candidates, labels))
                        random.shuffle(c)
                        candidates, labels = zip(*c)
                        ins = {
                            "db_id": db_id,
                            "question": question,
                            "candidates": candidates,
                            "labels": labels
                        }
                        output.append(ins)

    print(f"dev_count:{dev_count}")

    checker.export_sqlgen_miss(experiment_dir_name + '/' + SQL_MISS_FILE_NAME)
    checker.export_candidategen_miss(experiment_dir_name + '/' + CANDIDATE_MISS_FILE_NAME, num_total_count,
                                     candidate_num)

    return output,miss_candidates_log


if __name__ == "__main__":
    main()
    