import json
import os
from tqdm import tqdm
from collections import defaultdict
from dict_hash import dict_hash
from copy import deepcopy

from utils.evaluation.evaluate import Evaluator, build_foreign_key_map_from_json, rebuild_sql
from generator.QunitGenerator.utils import sql_nested_query_tmp_name_convert
from utils.spider_utils import fix_number_value, fix_query_toks_no_value

class RecallChecker():
    def __init__(self, dataset_file, tables_file, db_dir, output_dir):
        self.gold_sqls = defaultdict(set)
        self.gold_sql_dicts = defaultdict(list)
        self.sql_dict_map = defaultdict(dict)
        self.tables_file = tables_file
        self.db_dir = db_dir
        self.output_dir = output_dir
        self.kmaps = build_foreign_key_map_from_json(tables_file)
        self.evaluator = Evaluator()

        self.total_sqlgen_miss_count = 0
        self.total_sqlgen_miss_sqls = defaultdict(list)
        self.total_sqlgen_miss = defaultdict(dict)
        self.candidategen_miss_count = 0
        self.total_candidategen_miss_count = 0
        self.total_candidategen_miss_sqls = defaultdict(list)
        self.total_candidategen_miss = defaultdict(list)

        self.initialize(dataset_file, tables_file)

    def initialize(self, dataset_file, tables_file):
        print(f"Reading dataset : {dataset_file} for RecallChecker......")
        with open(dataset_file, "r") as data_file:
            data = json.load(data_file)
            for ex in tqdm(data):
                db_id = ex['db_id']
                # query = ex['query']
                ex = fix_number_value(ex)
                query = fix_query_toks_no_value(ex['query_toks_no_value'])
                g_sql = rebuild_sql(db_id, self.db_dir, sql_nested_query_tmp_name_convert(query), self.kmaps, tables_file)
                if g_sql not in self.gold_sql_dicts[db_id]:
                    self.gold_sqls[db_id].add(query)
                    self.gold_sql_dicts[db_id].append(g_sql)
                    self.sql_dict_map[db_id][dict_hash(g_sql)] = query

    def check_sqlgen_recall(self, db_id, sqls):
        gold_sqls = self.gold_sqls[db_id]
        gold_sql_dicts = self.gold_sql_dicts[db_id]
        gold_hit_sqls = set()

        gold_sql_num = len(gold_sqls)
        cnt = 0
        total_num = 0
        for sql in tqdm(sqls):
            total_num += 1
            p_sql = rebuild_sql(db_id, self.db_dir, sql_nested_query_tmp_name_convert(sql), self.kmaps, self.tables_file)
            for g_sql in gold_sql_dicts:
                if self.evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                    cnt += 1
                    print(f"\nHit      : {cnt:2<}/{gold_sql_num} @{total_num}")
                    gold_sql_dicts.remove(g_sql)
                    print(f"gold sql : {self.sql_dict_map[db_id][dict_hash(g_sql)]}")
                    gold_hit_sqls.add(self.sql_dict_map[db_id][dict_hash(g_sql)])
                    gold_sqls.remove(self.sql_dict_map[db_id][dict_hash(g_sql)])

        miss = gold_sql_num - cnt
        self.total_sqlgen_miss_count += miss
        self.total_sqlgen_miss_sqls[db_id] = list(gold_sqls)

        self.total_sqlgen_miss[db_id]['miss_rate'] = miss / gold_sql_num
        self.total_sqlgen_miss[db_id]['miss_count'] = miss
        self.total_sqlgen_miss[db_id]['miss_sql'] = list(gold_sqls)
        # ---------------------------------- FOR DEBUG ----------------------------------
        for gold_sql in gold_sqls:
            print(gold_sql)
        # ********************************** END DEBUG **********************************

        return miss / gold_sql_num

    def check_add_candidategen_miss(self, db_id, sqls, gold_sql, gold_dialect, gold_nl):
        g_sql = rebuild_sql(db_id, self.db_dir, sql_nested_query_tmp_name_convert(gold_sql), self.kmaps, self.tables_file)
        gold_sql_indices = []
        duplicate = -1
        for idx, candidate_sql in enumerate(sqls):
            candidate_sql = sql_nested_query_tmp_name_convert(candidate_sql)
            p_sql = rebuild_sql(db_id, self.db_dir, candidate_sql, self.kmaps, self.tables_file)
            if self.evaluator.eval_exact_match(deepcopy(p_sql), deepcopy(g_sql)) == 1:
                #print(f"gold_sql:{gold_sql}")
                #print(f"candidate_sql:{candidate_sql}")
                gold_sql_indices.append(idx)
                duplicate += 1
        #if duplicate > 0: print(f"duplicate:{duplicate}")         
        
        if not gold_sql_indices:
            self.total_candidategen_miss_count += 1
            #print(f"self.total_candidategen_miss_count:{self.total_candidategen_miss_count}")
            self.total_candidategen_miss_sqls[db_id].append(gold_sql)
            cur_miss = {}
            cur_miss['gold_sql'] = gold_sql
            cur_miss['gold_dialect'] = gold_dialect
            cur_miss['gold_nl'] = gold_nl

            if len(self.total_candidategen_miss[db_id]) == 0:
                cur_miss_count = {}
                cur_miss_count['stage1 miss count'] = 0
                cur_miss_count['stage2 miss count'] = 0
                self.total_candidategen_miss[db_id].append(cur_miss_count)
                
            if 'miss_sql' in self.total_sqlgen_miss[db_id].keys():
                if gold_sql in self.total_sqlgen_miss[db_id]['miss_sql']:
                    cur_miss['miss_stage'] = 1
                    self.total_candidategen_miss[db_id][0]['stage1 miss count'] += 1
                else:
                    self.candidategen_miss_count += 1
                    cur_miss['miss_stage'] = 2
                    self.total_candidategen_miss[db_id][0]['stage2 miss count'] += 1
            self.total_candidategen_miss[db_id].append(cur_miss)

        
        return gold_sql_indices

    def print_sqlgen_result(self, db_id, sql_generation_num):
        miss = len(self.total_sqlgen_miss_sqls[db_id])
        print(f"db :{db_id} sqlgen miss {miss}@{sql_generation_num}")

    def print_sqlgen_total_result(self, total_num, sql_generation_num):
        total = total_num
        miss = self.total_sqlgen_miss_count
        print(f"Total count:{total}; Total missing sql count: {miss}@{sql_generation_num}")
        print(f"SQL generation miss rate: {miss / total}")

    def print_candidategen_result(self, db_id, candidate_num):
        miss = len(self.total_candidategen_miss_sqls[db_id])
        print(f"db :{db_id} candidategen miss {miss}@{candidate_num}")

    def print_candidategen_total_result(self, total_num, candidate_num):
        total = total_num
        miss = self.total_candidategen_miss_count
        print(f"Total count:{total}; Total missing sql count: {miss}@{candidate_num}")
        print(f"Candidate generation miss rate: {miss / total}")

    def export_sqlgen_miss(self):
        filename  = os.path.join(self.output_dir, "sqlgen_miss_sqls.json")
        with open(filename, 'w') as f:
            f.write(f"Total sqlgen miss count:{self.total_sqlgen_miss_count}\n")
            json.dump(self.total_sqlgen_miss, f, indent=4)

    def export_candidategen_miss(self, total_num, candidate_num):
        total_miss = self.total_candidategen_miss_count
        miss = self.candidategen_miss_count
        filename = os.path.join(self.output_dir, "candidategen_miss_sqls.json")
        with open(filename, 'w') as f:
            f.write(f"Total test case count:{total_num}\n")
            f.write(f"Total miss count: {total_miss}\n")
            f.write(f"Total miss rate: {total_miss/total_num}\n")
            f.write(f"Stage2 missing count@candidate_num: {miss}@{candidate_num}\n")
            f.write(f"Candidate generation miss rate: {miss/total_num}\n")
            json.dump(self.total_candidategen_miss, f, indent=4)
            
