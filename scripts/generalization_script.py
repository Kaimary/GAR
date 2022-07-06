import os
import sys
import json
from tqdm import tqdm

from utils.recall_checker_utils import RecallChecker
from generator.QunitGenerator.qunit_generator import QunitSQLGenerator
from synthesizer.DialectSynthesizer.dialect_synthesizer import DialectSynthesizer
from configs.config import DIR_PATH, GENERATION_NUM, OVERWRITE_FLAG, REWRITE_FLAG, SERIALIZE_DATA_DIR

def serialization(
        dataset_name, data_file, tables_file, db_dir, output_dir):
    # Initialization
    sqls = []
    dialects = []
    pre_db_id = ""
    num_total_count = 0
    generator = QunitSQLGenerator(dataset_name, data_file, tables_file, db_dir, generation_num=GENERATION_NUM)
    synthesizer = DialectSynthesizer(tables_file, db_dir)
    # Use to check miss rate for each of the phase
    checker = RecallChecker(data_file, tables_file, db_dir, output_dir)

    serialization_dir = f'{DIR_PATH}{SERIALIZE_DATA_DIR.format(dataset_name)}'
    if not os.path.exists(serialization_dir): os.makedirs(serialization_dir)
    with open(data_file, "r") as data_file:
        data = json.load(data_file)
        for ex in data:
            num_total_count += 1
            db_id = ex['db_id']   
            
            # 进入每一个db时处理
            if pre_db_id == "" or pre_db_id != db_id:
                # Generate synthesis sql-dialects
                pre_db_id = db_id
                db_data_path = f'{serialization_dir}/{pre_db_id}_{GENERATION_NUM}.txt'
                if not os.path.exists(db_data_path) or OVERWRITE_FLAG:
                    print(f"db_id:{db_id}")
                    generator.switch_database(db_id)         
                    generator.switch_context()                                            
                    sqls = generator.generate()
                    for sql in sqls:
                        synthesizer.switch_database(db_id)
                        dialect = synthesizer.synthesize(sql)
                        dialects.append(dialect)

                    miss_rate = checker.check_sqlgen_recall(pre_db_id, sqls)
                    print(f"db: {db_id} miss rate:{miss_rate}@{GENERATION_NUM}")

                    datafile = open(db_data_path, 'w')
                    for sql, dialect in zip(sqls, dialects):
                        # write line to output file
                        line = sql + '\t' + dialect + '\n'
                        datafile.write(line)
                    datafile.close()
                    # Empty the lists
                    sqls.clear()
                    dialects.clear()
                # Rewrite dialects
                elif REWRITE_FLAG:
                    all_lines = []
                    datafile = open(db_data_path, 'r')
                    print(f"Rewrite dialects for database `{db_id}` ......")
                    for line in tqdm(datafile.readlines()):
                        sql, _ = line.split('\t')
                        synthesizer.switch_database(db_id)
                        dialect = synthesizer.synthesize(sql)
                        line = sql + '\t' + dialect + '\n'
                        all_lines.append(line)

                    datafile = open(db_data_path, 'w')
                    datafile.writelines(all_lines)
                    datafile.close()

    checker.print_sqlgen_total_result(num_total_count, GENERATION_NUM)
    checker.export_sqlgen_miss()
    return

dataset_name = 'spider'
tables_file = 'datasets/spider/tables.json'
db_dir = 'datasets/spider/database'
output_dir = 'saved_data/'
data_file = sys.argv[1]

serialization(dataset_name, data_file, tables_file, db_dir, output_dir)
