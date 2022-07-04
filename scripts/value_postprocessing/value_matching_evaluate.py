import json
import os
import click
from typing import List
from copy import deepcopy
from collections import defaultdict

from generator.QunitGenerator.utils import sql_nested_query_tmp_name_convert
from utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql
from configs.config import VALUE_FILTERED_TOPK_FILE_NAME

@click.command()
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("value_filtered_output_file", type=str)
@click.argument("output_file", type=click.Path(exists=True, dir_okay=True))
def main(tables_file, db_dir, value_filtered_output_file, output_file):
    pred_count = 0
    preds = []
    value_filtered_output_file = os.path.join(output_file, value_filtered_output_file)
    with open(value_filtered_output_file, "r") as pred_file:
        candidates = list()
        for pred in pred_file.readlines():
            if pred.strip():
                candidates.append(pred.strip())
            else:
                if candidates:
                    preds.append(candidates)
                    pred_count += 1
                candidates = list()

    miss_total = 0
    miss_pre = 0
    miss_stage3 = 0
    total_reranker_miss = defaultdict(list)
    total_pred_sqls = []
    evaluator = Evaluator()
    kmaps = build_foreign_key_map_from_json(tables_file)

    reranker_test_data_file = os.path.join(output_file, 'test.json')
    with open(reranker_test_data_file, "r") as reranker_file:
        data = json.load(reranker_file)
        for i, (ex, pred) in enumerate(zip(data, preds)):
            db_id = ex['db_id']
            labels = ex['labels']
            nl = ex['question']
            candidates = ex['candidates']
            candidate_sqls: List[str] = [sql.strip() for sql in ex['candidate_sqls']]
            top_sqls = []
            top_dialects = []
            top_sqls_index = map(lambda x: candidate_sqls.index(x), pred)
            for index in top_sqls_index:
                top_sqls.append(candidate_sqls[int(index)])
                top_dialects.append(candidates[int(index)])
            total_pred_sqls.append('\n'.join(top_sqls))

            if 1 not in labels:
                miss_pre += 1
                miss_total += 1
                if len(total_reranker_miss[db_id]) == 0:
                    total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 1, "stage3": 0})
                else:
                    total_reranker_miss[db_id][0]['miss'] += 1
                    total_reranker_miss[db_id][0]['stage1&2'] += 1
            else:
                flag = False
                match_pos = 0
                gold_pos = labels.index(1)
                gold_sql = candidate_sqls[gold_pos]
                g_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(gold_sql), kmaps, tables_file)

                for j, pred_sql in enumerate(top_sqls):
                    p_sql = rebuild_sql(db_id, db_dir, sql_nested_query_tmp_name_convert(pred_sql), kmaps, tables_file)

                    if evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                        flag = True
                        match_pos = j
                        break

                if not flag:
                    miss_total += 1
                    miss_stage3 += 1
                    if len(total_reranker_miss[db_id]) == 0:
                        total_reranker_miss[db_id].append({"miss": 1, "stage1&2": 0, "stage3": 1})
                    else:
                        total_reranker_miss[db_id][0]['miss'] += 1
                        total_reranker_miss[db_id][0]['stage3'] += 1

                    gold_dialect = candidates[gold_pos]
                    cur_miss = {
                        'nl': nl,
                        'gold_sql': gold_sql,
                        'gold_dialect': gold_dialect,
                        'pred_sql': top_sqls,
                        'pred_dialect': top_dialects
                    }
                    total_reranker_miss[db_id].append(cur_miss)

                else:
                    if len(total_reranker_miss[db_id]) == 0:
                        total_reranker_miss[db_id].append({"miss": 0, "stage1&2": 0, "stage3": 0})

                    if match_pos > 0:
                        gold_dialect = candidates[gold_pos]
                        cur_miss = {
                            'nl': nl,
                            'gold_sql': gold_sql,
                            'gold_dialect': gold_dialect,
                            'match_pos': match_pos + 1,
                            'pred_sql': top_sqls[0:match_pos + 1],
                            'pred_dialect': top_dialects[0:match_pos + 1]
                        }
                        total_reranker_miss[db_id].append(cur_miss)

    with open(output_file + '/' + VALUE_FILTERED_TOPK_FILE_NAME, 'w') as f:
        f.write(f"Total test case count:{pred_count}\n")
        f.write(f"Total miss count: {miss_total}\n")
        f.write(f"Total accuracy rate: {(pred_count - miss_total) / pred_count}\n")
        f.write(f"Stage1&2 missing count: {miss_pre}\n")
        f.write(f"Stage3 missing count: {miss_stage3}\n")
        f.write(f"Candidate generation miss rate: {miss_stage3 / pred_count}\n")
        json.dump(total_reranker_miss, f, indent=4)
    print(f"Value filter evaluation result saved in {output_file + '/' + VALUE_FILTERED_TOPK_FILE_NAME}")
    return


if __name__ == "__main__":
    main()
