import os
import json

from utils.recall_checker_utils import RecallChecker
from generator.QunitGenerator.qunit_generator import QunitSQLGenerator
from synthesizer.DialectSynthesizer.dialect_synthesizer import DialectSynthesizer
from configs.config import DIR_PATH, GENERATION_NUM, OVERWRITE_FLAG, SERIALIZE_DATA_DIR

def main():
    dataset_name = 'spider'
    tables_file = 'datasets/spider/tables.json'
    db_dir = 'datasets/spider/database'
    output_dir = 'saved_data/'
    # serialization(dataset_name, 'datasets/spider/gap_predicts.txt', tables_file, db_dir, output_dir)
    # serialization(dataset_name, 'sqlgenv2/datasets/geo/dev.json', tables_file, db_dir, output_dir)
    serialization(dataset_name, 'datasets/spider/dev_model_output.json', tables_file, db_dir, output_dir)

    return

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
                        datafile.write(sql)
                        datafile.write('\t')
                        datafile.write(dialect)
                        datafile.write("\n")
                    datafile.close()
                # Read from the existing file

    checker.print_sqlgen_total_result(num_total_count, GENERATION_NUM)
    checker.export_sqlgen_miss_sqls()
    return


if __name__ == "__main__":
    main()
