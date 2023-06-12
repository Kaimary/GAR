import os
import re
import time
import json
import random
import numpy as np
from typing import List
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from generator import abstract_generator
from configs.config import DIR_PATH, QUNITS_FILE, TIME_LIMIT_PRE_SQL
from generator.QunitGenerator.classes.rule_set import RULE_SET
from generator.QunitGenerator.classes.combinatorial_rule import COMBINATORIAL_RULE_DICTIONARY
from synthesizer.DialectSynthesizer.dialect_synthesizer import DialectSynthesizer
from generator.QunitGenerator.qunit.unit_extract import extract_spider_unit
from generator.QunitGenerator.utils import sql_nested_query_tmp_name_convert
from utils.evaluation.evaluate import build_foreign_key_map_from_json, rebuild_sql
from utils.evaluation.process_sql import get_schema, get_schema_from_json
from utils.spider_utils import DBSchema, get_all_schema, read_single_dataset_schema

class QunitSQLGenerator(abstract_generator.AbstractSQLGenerator):
    def __init__(self, dataset_name, data_file, tables_file, db_dir, generation_num):
        super().__init__(tables_file=tables_file, db_dir=db_dir)

        """
        :param tables_file: The database schemata file.
        :param db_dir: The directory of databases for the dataset.
        """
        self.dataset_name = dataset_name
        self.data_file = data_file
        self.generation_num = generation_num

        self.qunits = []
        self.syntax_constraint = {}
        self.skeletons = []
        self.rule_set = deepcopy(RULE_SET)
        self.synthesizer = DialectSynthesizer(tables_file=tables_file, db_dir=db_dir)
        self.combinatorial_rule_base = deepcopy(COMBINATORIAL_RULE_DICTIONARY)
        
        self.all_schema = get_all_schema(tables_file)
        self.kmaps = build_foreign_key_map_from_json(tables_file)
    
    def switch_context(self, *args):
        # Clear the existing context first
        self.qunits = []
        self.syntax_constraint = {}
        self.skeletons = []
        self.combinatorial_rule_base = deepcopy(COMBINATORIAL_RULE_DICTIONARY)
        
        # Read query units of the database
        if not os.path.exists(DIR_PATH + QUNITS_FILE.format(self.dataset_name)):
            print(
                f"Not found query units file! Please check if the file path setting is correct: "
                f"{QUNITS_FILE.format(self.dataset_name)}")
            print(f"Extract the query units...")
            extract_spider_unit(self.dataset_name, self.tables_file, self.data_file, db_path=self.db_dir)
            print(f"Extract the query units complete!")
        with open(DIR_PATH + QUNITS_FILE.format(self.dataset_name), "r") as data_file:
            qunit_set = json.load(data_file)
            self.syntax_constraint = qunit_set[self.db_name]['global_syntactic']
            skeletons = qunit_set[self.db_name]['skeleton']
            for skeleton in skeletons:
                parts = skeleton.split('|')
                arr = parts + ['"|"'] * (len(parts) - 1)
                self.skeletons.append(sorted(arr))
            for qn in qunit_set[self.db_name]['units']:
                new_qn = []
                new_qn.append(qn[0].replace("'", '"'))
                new_qn.extend(qn[1:])
                self.qunits.append(new_qn)

        if self.dataset_name == 'geo':
            self.rule_set.remove('nested_iue_rule')
        
        assert self.skeleton is [], "Please check the `qunits.json` and make sure the query units were extracted properly!"
        #  Determine the SQL skeleton types.
        for skeleton in self.combinatorial_rule_base["skeleton"].copy():
            exist = False
            if any(np.array_equal(sk, sorted(skeleton)) for sk in self.skeletons):
                exist = True
            if not exist:
                self.combinatorial_rule_base["skeleton"].remove(skeleton)
        # Update combinatorial rules based on the extracted query units.
        for qunit in self.qunits:
            if qunit[1] == "attr_select":
                self.combinatorial_rule_base["attr_select"].append(qunit)
            elif qunit[1] == "source":
                self.combinatorial_rule_base["source"].append(qunit)
            elif qunit[1] == "pred_where":
                self.combinatorial_rule_base["pred_where"].append(qunit)
            elif qunit[1] == "pred_where_subquery":
                self.combinatorial_rule_base["pred_where_subquery"].append(
                    qunit)
            elif qunit[1] == "attr_group":
                self.combinatorial_rule_base["attr_group"].append(qunit)
            elif qunit[1] == "pred_having":
                self.combinatorial_rule_base["pred_having"].append(qunit)
            elif qunit[1] == "attr_order":
                self.combinatorial_rule_base["attr_order"].append(qunit)
            elif qunit[1] == "argmax":
                self.combinatorial_rule_base["argmax"].append(qunit)
            elif qunit[1] == "argmin":
                self.combinatorial_rule_base["argmin"].append(qunit)
            elif qunit[1] == "iue":
                self.combinatorial_rule_base["iue"].append(qunit)
                
        return

    def switch_database(self, db_name):
        self.db_name = db_name
        self.db_schema = DBSchema(db_name, self.all_schema, db_path=self.db_dir)
        db_file = os.path.join(self.db_dir, db_name, db_name + ".sqlite")
        if not os.path.isfile(db_file):
            self.schema = get_schema_from_json(db_name, self.tables_file)
        else:
            self.schema = get_schema(db_file)
        _, self.table, self.table_dict = read_single_dataset_schema(self.tables_file, db_name)

    def generate(self):
        
        class RecursiveSyntaxConstraint:
            def __init__(self, syntax_constraint):
                self._syntax_constraint = syntax_constraint
                self.recursion_depth = defaultdict(int)
                self._init_recursion_depth()  # depth begin from 0

            def _init_recursion_depth(self):
                self.recursion_depth["attr_selects"] = self._syntax_constraint['max_projection_num']
                self.recursion_depth["attr_groups"] = self._syntax_constraint['max_group_num']
                self.recursion_depth["attr_orders"] = self._syntax_constraint['max_order_num']

                self.recursion_depth["where_conj"] = self._syntax_constraint['max_where_predicate_num'] - 1
                self.recursion_depth["having_conj"] = self._syntax_constraint['max_having_predicate_num'] - 1
                self.recursion_depth["pred_where_subquery"] = self._syntax_constraint['max_where_nested_num']

                self.recursion_depth["iues"] = self._syntax_constraint['max_iue_num']

        class RecursiveStateMachine:
            def __init__(self):
                self.info = defaultdict(list)
                self.depth = defaultdict(int)
                # Save generated fragments to avoid duplicate generation
                self.states = defaultdict(list)
                # It is used to stop attr_select recursion under having-IUE-SQL generation process.
                # In this case, we generate all the attr_selects in one-time instead of one-by-one.
                # Set the flag as True marks the "attr_select" recursion is over.
                self.attr_select_recursive_end = False

        class RecursiveCallbackHandler:
            def __init__(self, db_schema, syntax_constraint, combinatorial_rule_base):
                self.db_schema = db_schema
                self.constraint = RecursiveSyntaxConstraint(syntax_constraint)
                self.combinatorial_rule_base = combinatorial_rule_base

                self.begin_function_map = defaultdict(
                    lambda: self._default_begin_handler)
                self.choice_function_map = defaultdict(
                    lambda: self._default_choice)
                self.end_function_map = defaultdict(
                    lambda: self._default_end_handler)

                # self._init_random_terminal_handler()
                self._init_end_handler()
                self._init_choice()

            def add_table_ref(self, candidate, dep, source):
                # If star-related attribute or single-relation SQL, no need for processing
                if '*' in candidate or "JOIN " not in source:
                    return candidate
                # For without-using-alias join SQL.
                if '.' in candidate:
                    return candidate

                # Split up candidate into two parts,
                # to avoid the confusion that may cause by predicate fragment.
                part1 = candidate
                part2 = ""
                npos = candidate.find('( SELECT')
                if npos != -1:
                    part1 = candidate[:npos]
                    part2 = candidate[npos:]
                tokens = part1.strip('"').split(" ")
                pos = 0
                pos_disnt = -1
                pos_agg = -1
                if 'DISTINCT' in tokens:
                    # In Scolar dataset, it is possible to have more than one DISTINCT in a unit.
                    # We index the last DISTINCT instead.
                    pos_disnt = len(tokens) - list(reversed(tokens)).index('DISTINCT') - 1
                for idx, token in enumerate(tokens):
                    if token.strip() in ['COUNT', 'MAX', 'MIN', 'SUM', 'AVG']:
                        pos_agg = idx

                if pos_disnt > -1 or pos_agg > -1:
                    if pos_disnt > pos_agg:
                        pos = pos_disnt + 1
                    else:
                        pos = pos_agg + 2

                tokens[pos] = dep + '.' + tokens[pos]

                result = '"' + ' '.join(tokens) + part2 + '"'
                return result

            def replace_op_placeholder(self, candidate, source):
                if '__OP__' not in candidate:
                    return candidate

                # Determine the type of the associated column for the operator
                tokens = candidate.strip('"').split(" ")
                opt_pos = tokens.index('__OP__')
                col = tokens[opt_pos - 1]
                if col == ')':
                    col_type = "NUM"
                elif '.' in col:
                    table = col.split('.')[0].lower()
                    col_name = col.split('.')[1].lower()
                    col_type = "NUM" if col_name in self.db_schema.table_ncolumn[table] else "STR"
                else:
                    table = source.strip('"').lower()
                    col_name = col.lower()
                    col_type = "NUM" if col_name in self.db_schema.table_ncolumn[table] else "STR"
                # Random choice the operator based on the column type
                if col_type == "NUM":
                    op = random.choices([">", "<", ">=", "<=", "=", "!="], weights=(
                        17, 17, 17, 17, 17, 15))[0]
                else:
                    op = random.choices(
                        ["=", "!=", "LIKE"], weights=(60, 30, 10))[0]
                tokens[opt_pos] = op

                result = '"' + ' '.join(tokens) + '"'
                return result

            def not_satisfy_depth_constraint(self, node_name):

                if node_name in self.constraint.recursion_depth:
                    return state_machine.depth[node_name] >= self.constraint.recursion_depth[node_name]
                return False

            def choice(self, node_name, state_machine):
                return self.choice_function_map[node_name](node_name, state_machine)

            def begin_handler(self, node_name, state_machine):
                self.begin_function_map[node_name](node_name, state_machine)

            def end_handler(self, node_name, ret, state_machine):
                self.end_function_map[node_name](node_name, ret, state_machine)

            def _default_begin_handler(self, node_name, state_machine):
                state_machine.depth[node_name] += 1

            def _default_end_handler(self, node_name, ret, state_machine):
                if node_name != "pred_where_subquery":
                    state_machine.depth[node_name] -= 1
                state_machine.info[node_name].append(ret)

            def _default_choice(self, node_name, state_machine):
                if not self.combinatorial_rule_base[node_name]:
                    return None

                ret_i = random.randrange(
                    len(self.combinatorial_rule_base[node_name]))
                return self.combinatorial_rule_base[node_name][ret_i]

            def _init_end_handler(self):
                None

            def _init_choice(self):
                self.choice_function_map['source'] = self._source_choice
                self.choice_function_map['argmin'] = self._attr_pred_choice
                self.choice_function_map['argmax'] = self._attr_pred_choice
                self.choice_function_map['attr_select'] = self._attr_pred_choice
                self.choice_function_map['attr_group'] = self._attr_pred_choice
                self.choice_function_map['attr_order'] = self._attr_pred_choice
                self.choice_function_map['pred_where'] = self._attr_pred_choice
                self.choice_function_map['pred_where_subquery'] = self._attr_pred_choice
                self.choice_function_map['pred_having'] = self._attr_pred_choice
                self.choice_function_map['iue'] = self._iue_choice

            def _source_choice(self, node_name, state_machine):
                if state_machine.states["iue"]:
                    tables = [col.split('.')[0] if 'DISTINCT' not in col else col.split('.')[0][8:]
                              for col in state_machine.states["iue"]]
                    sources = [qunit[0] for qunit in self.combinatorial_rule_base[node_name] if all(
                        tbl in qunit[0] for tbl in tables)]
                    weights = [qunit[3] for qunit in self.combinatorial_rule_base[node_name] if all(
                        tbl in qunit[0] for tbl in tables)]
                else:
                    sources = [qunit[0]
                               for qunit in self.combinatorial_rule_base[node_name]]
                    weights = [qunit[3]
                               for qunit in self.combinatorial_rule_base[node_name]]

                if not sources:
                    return None

                candidate = random.choices(sources, weights=weights, k=3)[0]
                return [candidate]

            def _attr_pred_choice(self, node_name, state_machine):
                if not self.combinatorial_rule_base[node_name]:
                    return None

                # if IUE exists, we select out the columns from state machine directly.
                if state_machine.states["iue"] and node_name == "attr_select":
                    if state_machine.attr_select_recursive_end:
                        return None

                    attributes = ','.join(state_machine.states["iue"])
                    # Hard code to fix the table reference on asterisk case
                    if "*" in state_machine.states["iue"][0]:
                        attributes = "*"
                    candidate = '"' + attributes + '"'
                    state_machine.attr_select_recursive_end = True
                else:
                    source_info = state_machine.info['source'][0]
                    source_tokens = source_info.strip('"').split(" ")
                    # Before random choice, we use the dependency of each query unit to make sure it fits on the state machine.
                    candidates = []
                    weights = []
                    deps = []
                    for qunit in self.combinatorial_rule_base[node_name]:
                        # Get the dependency of this qunit
                        dep = qunit[2][0]
                        # We consider two scenarios as fitting state machine:
                        # 1. Exact match. e.g. 1) "table1" == "table1"; 2) "table1 as t1 JOIN table2 as t2" == "table1 as t1 JOIN table2 as t2"
                        # 2. Existing match e.g. 1) "table1" == "table1 as t1 JOIN table2 as t2"
                        if dep == source_info.strip('"') or dep in source_tokens:
                            # Check if it has been generated
                            # Except the evaluation on "pred_where" (e.g. YEAR = 2014 OR YEAR = 2015)
                            if (qunit[1] == "pred_where" and '=' in qunit[0].split()) or \
                                    qunit[0] not in state_machine.states[qunit[1]]:
                                candidates.append(qunit[0])
                                weights.append(qunit[3])
                                deps.append(dep)

                    if not candidates:
                        return None

                    ret_list = list(range(len(candidates)))
                    ret_i = random.choices(ret_list, weights=weights, k=3)[0]
                    # Add into state machine
                    category = self.combinatorial_rule_base[node_name][0][1]
                    state_machine.states[category].append(candidates[ret_i])
                    # Add alias if needed.
                    # if node_name == "pred_where_subquery":
                    #     print("STOP")
                    candidate = self.add_table_ref(
                        candidates[ret_i], deps[ret_i], source_info)
                    candidate = self.replace_op_placeholder(
                        candidate, source_info)
                    # Special case for argmin/argmax
                    # limit_num = random.choices([1,2,3,5,8,10], weights=[1167, 4, 89, 23, 2, 12], k=3)[0]
                    limit_num = 1
                    if node_name == "argmin":
                        if random.random() < 0.5:
                            candidate = '"ORDER BY ' + \
                                        candidate.strip('"') + ' ASC LIMIT ' + \
                                        str(limit_num) + '"'
                        else:
                            candidate = '"ORDER BY ' + \
                                        candidate.strip('"') + ' LIMIT ' + \
                                        str(limit_num) + '"'
                    if node_name == "argmax":
                        candidate = '"ORDER BY ' + \
                                    candidate.strip('"') + ' DESC LIMIT ' + \
                                    str(limit_num) + '"'

                return [candidate]

            def _iue_choice(self, node_name, state_machine):

                if not self.combinatorial_rule_base[node_name]:
                    return None

                weights = [qunit[3]
                           for qunit in self.combinatorial_rule_base[node_name]]
                ret_list = list(
                    range(len(self.combinatorial_rule_base[node_name])))
                ret_i = random.choices(ret_list, weights=weights, k=3)[0]

                exp = self.combinatorial_rule_base[node_name][ret_i][0]
                cat = self.combinatorial_rule_base[node_name][ret_i][1]
                dep = self.combinatorial_rule_base[node_name][ret_i][2]
                # We determine projections(and sources) based on the dependencies in iue.
                # 1. pk-fk-related columns with dependencies; (REMOVE)
                # 2. The same with dependencies.
                sel_col_candidates = [dep]
                # pk_fk_cols = []
                # for s_col_full in dep:
                #     for fk, pk in self.db_schema.table_foreign_key.items():
                #         if s_col_full == fk.upper():
                #             pk_fk_cols.append(pk.upper())
                #         elif s_col_full == pk.upper():
                #             pk_fk_cols.append(fk.upper())
                # if len(pk_fk_cols) == len(dep):
                #     sel_col_candidates.append(pk_fk_cols)

                sel_cols = random.choice(sel_col_candidates)
                state_machine.states[cat].extend(sel_cols)

                return [exp]

            # DEPRECATED
            # def _iue_choice(self, node_name, state_machine):
            #     attr_selects = state_machine.info['attr_select']
            #     # We assume that all aggregate or asterisk related attribute will not be along with IUE.
            #     if any(any(k in sel for k in ['*', 'COUNT', 'AVG', 'MAX', 'MIN', 'SUM']) for sel in attr_selects):
            #         return None

            #     if all('.' not in sel for sel in attr_selects):
            #         source = state_machine.info['source'][0].strip('"')
            #         attr_selects = [
            #             f'{source}.{sel[1:-1]}' for sel in attr_selects]
            #     # Before random choice, we use the dependency of each query unit to make sure it fits on the state machine.
            #     candidates = [qunit[0] for qunit in self.combinatorial_rule_base[node_name]
            #                   if sorted(attr_selects) == sorted(qunit[2])]
            #     weights = [qunit[3] for qunit in self.combinatorial_rule_base[node_name]
            #                if sorted(attr_selects) == sorted(qunit[2])]
            #     if not candidates:
            #         return None

            #     candidate = random.choices(candidates, weights=weights, k=3)[0]
            #     return [candidate]

        handler = RecursiveCallbackHandler(
            self.db_schema, self.syntax_constraint, self.combinatorial_rule_base)

        def dfs_random(node_name: str, state_machine: RecursiveStateMachine) -> List[str]:
            # print(f"node_name: {node_name}")
            ret = ""
            if handler.not_satisfy_depth_constraint(node_name):
                return None

            handler.begin_handler(node_name, state_machine)

            expression = handler.choice(node_name, state_machine)
            if not expression:
                return None

            for element in expression:
                if '"' in element:  # ternimal
                    ret += element.strip('"')
                else:  # non-ternimal
                    tmp_str = dfs_random(element, state_machine)
                    if tmp_str == None:
                        return None
                    ret += tmp_str

            handler.end_handler(node_name, element, state_machine)

            return ret

        def reorder_sql(sql) -> str:
            # We first reorder SQL clauses to make it align with the grammar;
            fragments = sql.split('|')
            s_prj = ""
            s_src = ""
            s_group = ""
            s_where = ""
            others = []
            for fragment in fragments:
                if 'SELECT ' in fragment and all(
                        key not in fragment for key in ['( SELECT', 'INTERSECT', 'UNION', 'EXCEPT']):
                    s_prj = fragment
                elif 'FROM ' in fragment and not s_src and all(
                        key not in fragment for key in ['( SELECT', 'INTERSECT', 'UNION', 'EXCEPT']):
                    s_src = fragment
                elif 'GROUP BY ' in fragment and all(
                        key not in fragment for key in ['( SELECT', 'INTERSECT', 'UNION', 'EXCEPT']):
                    s_group = fragment
                elif 'WHERE ' in fragment and all(key not in fragment for key in ['INTERSECT', 'UNION', 'EXCEPT']):
                    s_where = fragment
                else:
                    others.append(fragment)

            # Then for projection and selection, we reorder columns and predicates in asending order.
            projections = s_prj[6:].split(',')
            projections = sorted(projections)
            s_prj = 'SELECT ' + ','.join(projections)

            if s_where:
                # Currently we reorder selection if and only if one type condition(AND/OR) existis.
                # TODO Mixed conditions support
                mix_cond = False

                where_toks = s_where[6:].split(' ')
                cond = ''
                predicates = []
                inside_bracket = 0
                inside_between = 0
                start = 0
                for i, tok in enumerate(where_toks):
                    if '(' in tok:
                        inside_bracket += 1
                    elif ')' in tok:
                        inside_bracket -= 1
                    elif 'BETWEEN' in tok:
                        inside_between += 1
                    elif inside_between > 0 and 'AND' in tok:
                        inside_between -= 1
                    elif tok in ['AND', 'OR'] and inside_bracket == 0:
                        if not cond or cond == tok:
                            cond = tok.strip()
                            predicate = ' '.join(where_toks[start:i])
                            predicates.append(predicate)
                            start = i + 1
                        else:
                            mix_cond = True
                            break
                if not mix_cond and predicates:
                    last_predicate = ' '.join(where_toks[start:len(where_toks)])
                    predicates.append(last_predicate)
                    predicates = sorted(predicates)
                    if cond == 'AND':
                        s_where = 'WHERE ' + ' AND '.join(predicates)
                    else:
                        s_where = 'WHERE ' + ' OR '.join(predicates)

            result = s_prj + " " + s_src + " " + s_where + \
                     " " + s_group + " " + " ".join(others)
            return result

        dialect_sql_dict = defaultdict(int)
        index = 0
        sqls = []
        sql_map_set = set()
        timeout = False
        for i in tqdm(range(self.generation_num)):
            time_start = time.time()
            if timeout:
                break
            # if i % 1000 == 0:
            #     random.seed(datetime.datetime.now().microsecond)
            ret = None
            while True:
                state_machine = RecursiveStateMachine()
                ret = dfs_random("skeleton", state_machine)
                if ret:
                    # print(state_machine.tables)
                    is_valid = True
                    # We reorder the SQL str to make it align with SQL grammar.
                    # print(f"ret:{ret}")
                    sql = reorder_sql(ret)
                    sql = re.sub(r"(?<=\s)VALUE", "'VALUE'", sql)
                    # sql = sql.replace('VALUE', "'VALUE'")
                    # print(f"sql:{sql}")
                    # We define a set of global rules to post-check the validity of SQL.
                    # (TESTING PURPSE) sql = "select MAX ( CAPACITY ) , CAPACITY from STADIUM"
                    try:
                        p_sql = rebuild_sql(
                            self.db_name, self.db_dir, sql_nested_query_tmp_name_convert(sql), self.kmaps,
                            self.tables_file)
                        p_sql['primary_cols'] = self.db_schema.primary_cols

                    except AssertionError:
                        continue
                    for r in self.rule_set:
                        res = r.apply_rule(p_sql)
                        if not res:
                            is_valid = False
                            break
                    if 'primary_cols' in p_sql:
                        p_sql.pop('primary_cols')

                    if is_valid:
                        time_end = time.time()
                        timeout = (time_end - time_start) > TIME_LIMIT_PRE_SQL
                        sql_map = json.dumps(p_sql)
                        if sql_map in sql_map_set:
                            if timeout:
                                break
                            else:
                                continue
                        else:
                            sql = sql.replace('  ', ' ')
                            sql_map_set.add(sql_map)
                            # We use dialect as the key to check if current SQL has the same dialect ever.
                            # If so, we will compare and use the shortest one.
                            self.synthesizer.switch_database(self.db_name)
                            # print(f"sql:{sql}")
                            dialect = self.synthesizer.synthesize(sql)
                            if dialect not in dialect_sql_dict.keys():
                                dialect_sql_dict[dialect] = index
                                sqls.append(sql)
                                index += 1
                                break
                            else:
                                prev_i = dialect_sql_dict[dialect]
                                prev_sql = sqls[prev_i]
                                if len(prev_sql) > len(sql):
                                    # dialect_sql_dict[dialect] = index
                                    sqls[prev_i] = sql
                                continue

        return sqls
