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
    RERANKER_TRAIN_DATA_FILE, RERANKER_DEV_DATA_FILE, REWRITE
from spider_utils.recall_checker_utils import RecallChecker


@click.command()
@click.argument("dataset_name", default="spider")
@click.argument("train_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("dev_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("num_sql", default=100)
@click.argument("overwrite", default=0)
@click.argument("sentence_embedder_model_dir", default="")
def main(dataset_name, train_file, dev_file, tables_file, db_dir, num_sql, overwrite, sentence_embedder_model_dir):
    embedder = SentenceTransformer(sentence_embedder_model_dir)
    kmaps = build_foreign_key_map_from_json(tables_file)

    output_train_data = generate_rerank_data_from_spider_dataset(embedder, kmaps, dataset_name, train_file, tables_file, db_dir,
                                                                    sql_generation_num=num_sql, overwrite=overwrite)
    output_dev_data = generate_rerank_data_from_spider_dataset(embedder, kmaps, dataset_name, dev_file, tables_file, db_dir,
                                                                sql_generation_num=num_sql, dev=True,
                                                                max_instances=1000, candidate_num=100,
                                                                overwrite=overwrite)

    print(f"Overwrite re-ranking data into file: {RERANKER_TRAIN_DATA_FILE.format(dataset_name)} and {RERANKER_DEV_DATA_FILE.format(dataset_name)}")
    with open(DIR_PATH + RERANKER_TRAIN_DATA_FILE.format(dataset_name), 'w') as outfile:
        json.dump(output_train_data, outfile, indent=4)
    with open(DIR_PATH + RERANKER_DEV_DATA_FILE.format(dataset_name), 'w') as outfile:
        json.dump(output_dev_data, outfile, indent=4)
    print(f"Overwrite re-ranking data succeed!")

    return


def generate_rerank_data_from_spider_dataset(
        embedder, kmaps, dataset_name, dataset_file, tables_file, db_dir, dev=False, max_instances=None, sql_generation_num=10000,
        candidate_num=100, overwrite=0
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

    with open(dataset_file, "r") as data_file:
        data = json.load(data_file)
        for ex in data:
            num_total_count += 1
            db_id = ex['db_id']
            
            if db_id == 'company_1' or db_id == 'baseball_1' or db_id == 'customer_complaints': 
                continue
            
            ex = fix_number_value(ex)
            gold_sql = fix_query_toks_no_value(ex['query_toks_no_value'])

            question = ex['question']
            if not current_db_id or current_db_id != db_id:
                if current_db_id: checker.print_candidategen_result(current_db_id, candidate_num)

                current_db_id = db_id
                print(f"db_id: {current_db_id}")
                db_count += 1
                print(f"db_count: {db_count}")
                sqls.clear()
                dialects.clear()
                # We firstly strore all gold sqls in current db into the full set.
                sql_map_set = set()
                db_file = os.path.join(db_dir, current_db_id, current_db_id + ".sqlite")
                if not os.path.isfile(db_file):
                    schema = get_schema_from_json(db_id, tables_file)
                else:
                    schema = get_schema(db_file)
                _, table, table_dict = read_single_dataset_schema(tables_file, current_db_id)
                for ex1 in data:
                    if ex1['db_id'] == current_db_id:
                        ex1 = fix_number_value(ex1)
                        # Reconstruct gold sql by masking out value tokens
                        gold_sql1 = fix_query_toks_no_value(ex1['query_toks_no_value'])
                        sql_map = json.dumps(
                            rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(gold_sql1), kmaps))
                        if sql_map in sql_map_set:
                            continue
                        else:
                            sql_map_set.add(sql_map)
                            sqls.append(gold_sql1)
                            gold_sql1 = sql_nested_query_tmp_name_convert(gold_sql1)
                            _, sql_dict, schema_ = disambiguate_items2(tokenize(gold_sql1), schema, table,
                                                                       allow_aliases=False)
                            dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                            dialects.append(dialect)

                db_data_path = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}/{current_db_id}_{sql_generation_num}.txt'
                if not os.path.exists(db_data_path) or bool(overwrite):
                    db_schema = DBSchema(current_db_id, all_schema)
                    g = Generator(dataset_name, current_db_id, db_schema, dataset_file, tables_file, db_dir, kmaps)

                    # def generator():
                    #     while len(sqls) < sql_generation_num:
                    #         yield

                    # for _ in tqdm(generator()): 
                    sqls = g.generate_sql(sql_generation_num)
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
                            _, sql_dict, schema_ = disambiguate_items2(tokenize(sql1), schema, table, allow_aliases=False)
                            dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                            line = sql + '\t' + dialect + '\n'
                            all_lines.append(line)

                        dialects.append(dialect.strip())
                    datafile.close()

                    if REWRITE == True:
                        datafile = open(db_data_path, 'w')
                        datafile.writelines(all_lines)
                        datafile.close()
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
            # print(dialects[indices[0,idx]], "(Distance: %.4f)" % distances[0,idx])

            # Validate if gold sql included in the candidates
            # start_time = datetime.datetime.now()
            # evaluator = Evaluator()
            # kmaps = build_foreign_key_map_from_json(tables_file)
            # g_sql = rebuild_sql(current_db_id, db_dir, gold_sql, kmaps)
            # gold_sql_index = -1
            # for idx, candidate_sql in enumerate(candidate_sqls):
            #     candidate_sql = sql_nested_query_tmp_name_convert(candidate_sql)
            #     p_sql = rebuild_sql(current_db_id, db_dir, candidate_sql, kmaps)
            #     if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
            #         gold_sql_index = idx
            #         break
            # end_time = datetime.datetime.now()
            # print(f"Validate gold SQL complete: {end_time-start_time}")
            # Add the gold sql into candidates if not exists
            gold_sql_indices = checker.check_add_candidategen_miss_sql(db_id, candidate_sqls, gold_sql, db_dir, kmaps)
            if not gold_sql_indices:
                candidate_sqls.append(gold_sql)
                gold_sql = sql_nested_query_tmp_name_convert(gold_sql)
                _, sql_dict, schema_ = disambiguate_items2(tokenize(gold_sql), schema, table, allow_aliases=False)
                gold_sql_dialect = convert_sql_to_dialect(sql_dict, table_dict, schema_)
                candidate_dialects.append(gold_sql_dialect)
                gold_sql_indices.append(candidate_num)
            # Otherwise, add one more candidate from the whole set
            else:
                # gold_sql_index = candidate_sqls.index(gold_sql)
                candidate_sqls.append(sqls[indices[0, candidate_num]])
                candidate_dialects.append(dialects[indices[0, candidate_num]])
                # # Swap to make sure that the gold one is the last element.
                # candidate_sqls[gold_sql_index], candidate_sqls[candidate_num] = candidate_sqls[candidate_num], \
                #                                                                 candidate_sqls[gold_sql_index]
                # candidate_dialects[gold_sql_index], candidate_dialects[candidate_num] = candidate_dialects[
                #                                                                             candidate_num], \
                #                                                                         candidate_dialects[
                #                                                                             gold_sql_index]

            if dev and max_instances and dev_count < max_instances:
                coin = random.choices([1, 0], weights=(50, 50))[0]
                if bool(coin):
                    dev_count += 1
                    candidates = candidate_dialects
                    labels = [1 if i in gold_sql_indices else 0 for i in range(candidate_num+1)] 
                    # labels = [0] * candidate_num + [1]
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
            else:
                for j in range(0, (int)(candidate_num / 10)):
                    coin = random.choices([1, 0], weights=(30, 70))[0]
                    if bool(coin):
                        start = j * 10
                        end = (j + 1) * 10
                        candidates = candidate_dialects[start: end]
                        labels = [1 if i in gold_sql_indices else 0 for i in range(start, end)]
                        if 1 not in labels:
                            candidates.append(candidate_dialects[gold_sql_indices[0]])
                            labels.append(1)
                        else:
                            ii = random.choice(range(candidate_num+1))
                            candidates.append(candidate_dialects[ii])
                            if ii in gold_sql_indices: 
                                labels.append(1)
                            else:
                                labels.append(0)
                        # labels = [0] * 10 + [1]
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

    checker.print_candidategen_total_result(num_total_count, candidate_num)
    checker.export_candidategen_miss_sqls()
    return output


if __name__ == "__main__":
    main()
