import os
import random
import click
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.recall_checker_utils import RecallChecker
from utils.spider_utils import fix_number_value,fix_query_toks_no_value
from generator.QunitGenerator.qunit_generator import QunitSQLGenerator
from synthesizer.DialectSynthesizer.dialect_synthesizer import DialectSynthesizer
from configs.config import DIR_PATH, RERANKER_DEV_DATA_FILE, RERANKER_TRAIN_DATA_FILE, SERIALIZE_DATA_DIR, GENERATION_NUM, CANDIDATE_NUM, \
    REWRITE_FLAG, OVERWRITE_FLAG, RERANKER_DEV_DATA_MAX_NUM, \
    RETRIEVAL_MODEL_DIR, RETRIEVAL_MODEL_NAME, RETRIEVAL_MODEL_DIMENSION, \
    SQLGEN_DEBUG_FLAG, RETREVAL_DEBUG_FLAG


@click.command()
@click.argument("dataset_name", default="spider")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("mode", default="test")
def main(dataset_name, dataset_file, tables_file, db_dir, output_dir, mode):
    # Initialization
    output = []
    total = 0
    db_count = 0
    current_db_id = ""
    sqls = []
    dialects = []
    miss_candidates_log = []

    retrieval_model = SentenceTransformer(DIR_PATH + RETRIEVAL_MODEL_DIR.format(dataset_name) + '/' + RETRIEVAL_MODEL_NAME)
    generator = QunitSQLGenerator(dataset_name, dataset_file, tables_file, db_dir, generation_num=GENERATION_NUM)
    synthesizer = DialectSynthesizer(tables_file, db_dir)
    checker = RecallChecker(dataset_file, tables_file, db_dir, output_dir)

    with open(dataset_file, "r") as data_file:
        data = json.load(data_file)
        for i, ex in enumerate(data):
            total += 1
            db_id = ex['db_id']
            # Exclude below databases for code defects
            if db_id == 'company_1' or db_id == 'baseball_1' or db_id == 'customer_complaints': continue
            ex = fix_number_value(ex)
            question = ex['question']
            gold_sql = fix_query_toks_no_value(ex['query_toks_no_value'])
            # Get gold dialect
            synthesizer.switch_database(db_id)
            gold_dialect = synthesizer.synthesize(gold_sql)

            if not current_db_id or current_db_id != db_id:
                if current_db_id:
                    if RETREVAL_DEBUG_FLAG: checker.print_candidategen_result(current_db_id, CANDIDATE_NUM)

                current_db_id = db_id
                print(f"db_id: {current_db_id}")
                db_count += 1
                # print(f"db_count: {db_count}")
                sqls.clear()
                dialects.clear()

                db_data_path = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}/{current_db_id}_{GENERATION_NUM}.txt'
                if not os.path.exists(db_data_path) or OVERWRITE_FLAG:
                    # for _ in tqdm(generator()): 
                    generator.switch_database(current_db_id)
                    generator.switch_context()
                    sqls = generator.generate()
                    sqls = list(sqls)
                    
                    for sql in sqls:
                        synthesizer.switch_database(current_db_id)
                        dialect = synthesizer.synthesize(sql)
                        dialects.append(dialect)
                        
                    datafile = open(db_data_path, 'w')
                    for sql, dialect in zip(sqls, dialects):
                        # write line to output file
                        line = sql + '\t' + dialect + '\n'
                        datafile.write(line)
                    datafile.close()
                # Read from the existing file
                else:
                    all_lines = []
                    datafile = open(db_data_path, 'r')
                    for line in datafile.readlines():
                        sql, dialect = line.split('\t')
                        sqls.append(sql.strip())

                        if REWRITE_FLAG:
                            synthesizer.switch_database(current_db_id)
                            dialect = synthesizer.synthesize(sql)
                            line = sql + '\t' + dialect + '\n'
                            all_lines.append(line)

                        dialects.append(dialect.strip())
                    datafile.close()

                    if REWRITE_FLAG:
                        datafile = open(db_data_path, 'w')
                        datafile.writelines(all_lines)
                        datafile.close()

                if SQLGEN_DEBUG_FLAG: checker.check_sqlgen_recall(db_id, sqls)

                dialect_embeddings = retrieval_model.encode(dialects)
                index = faiss.IndexFlatL2(int(RETRIEVAL_MODEL_DIMENSION))
                index.add(np.stack(dialect_embeddings, axis=0))

            question_embedding = retrieval_model.encode(question)
            _, indices = index.search(np.asarray(question_embedding).reshape(1, int(RETRIEVAL_MODEL_DIMENSION)), CANDIDATE_NUM + 1)
            candidate_dialects = [dialects[indices[0, idx]] for idx in range(0, CANDIDATE_NUM)]
            candidate_sqls = [sqls[indices[0, idx]] for idx in range(0, CANDIDATE_NUM)]

            gold_sql_indices = checker.check_add_candidategen_miss(
                db_id, candidate_sqls, gold_sql, gold_dialect, question
            )
            # For training purpose, make sure all the instances include the gold query
            if mode in ['train', 'dev']:
                add_sql = sqls[indices[0, CANDIDATE_NUM]]
                add_dialect = dialects[indices[0, CANDIDATE_NUM]]
                if not gold_sql_indices:
                    add_sql = gold_sql
                    synthesizer.switch_database(current_db_id)
                    add_dialect = synthesizer.synthesize(gold_sql)
                    gold_sql_indices.append(CANDIDATE_NUM)
                candidate_sqls.append(add_sql)
                candidate_dialects.append(add_dialect)
            elif mode=='test' and gold_sql_indices:
                candidate_sqls[gold_sql_indices[-1]], candidate_sqls[CANDIDATE_NUM - 1] = \
                    candidate_sqls[CANDIDATE_NUM - 1], candidate_sqls[gold_sql_indices[-1]]
                candidate_dialects[gold_sql_indices[-1]], candidate_dialects[CANDIDATE_NUM - 1] = \
                    candidate_dialects[CANDIDATE_NUM - 1], candidate_dialects[gold_sql_indices[-1]]
    
            ins = {
                "index": i,
                "db_id": db_id,
                "question": question
            }
            if mode == 'train' or (mode=='dev' and len(output) < RERANKER_DEV_DATA_MAX_NUM):
                coin = random.choices([1, 0], weights=(50, 50))[0]
                if bool(coin):
                    candidates = candidate_dialects
                    labels = [1 if i in gold_sql_indices else 0 for i in range(CANDIDATE_NUM+1)] 
                    # labels = [0] * candidate_num + [1]
                    # Shuffle the list
                    c = list(zip(candidates, labels))
                    random.shuffle(c)
                    candidates, labels = zip(*c)
                    ins["candidates"] = list(candidates)
                    ins["labels"] = list(labels)
            else:
                candidates = candidate_dialects
                if gold_sql_indices:
                    labels = [0] * (CANDIDATE_NUM - 1) + [1]
                else:
                    labels = [0] * (CANDIDATE_NUM - 1) + [0]
                    miss_candidates_log.append({
                        'num': total,
                        'question': question,
                        'gold_dialect': gold_dialect,
                        'gold_sql': gold_sql,
                        'top 10 predict dialect': [dia for dia in candidates[:10]],
                        'top 10 predict sql': [sql for sql in candidate_sqls[:10]]
                    })
                ins["candidates"] = candidates
                ins["candidate_sqls"] = candidate_sqls
                ins["labels"] = labels

            if "candidates" in ins.keys(): output.append(ins)

    # *Debug purpose*
    if SQLGEN_DEBUG_FLAG: checker.export_sqlgen_miss()
    if RETREVAL_DEBUG_FLAG: checker.export_candidategen_miss(total, CANDIDATE_NUM)

    if mode == 'train': output_file = DIR_PATH + RERANKER_TRAIN_DATA_FILE.format(dataset_name)
    elif mode == 'dev': output_file = DIR_PATH + RERANKER_DEV_DATA_FILE.format(dataset_name)
    else: output_file = os.path.join(output_dir, f'{mode}.json')
    debug_file = os.path.join(output_dir, 'miss_candidate_log.json')
    with open(output_file, 'w') as outfile:
        json.dump(output, outfile, indent=4)
    with open(debug_file, 'w') as outfile:
        json.dump(miss_candidates_log, outfile, indent=4)

    return

if __name__ == "__main__":
    main()
    # dataset_file = 'datasets/spider/dev.json'
    # main(
    #     'spider',
    #     dataset_file, 
    #     'datasets/spider/tables.json', 
    #     'datasets/spider/database', 
    #     mode='test'
    # )