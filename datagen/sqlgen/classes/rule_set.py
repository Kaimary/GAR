from collections import defaultdict
from typing import Dict

from spider_utils.evaluation.evaluate import rebuild_sql
from ..utils.sql_tmp_update import sql_nested_query_tmp_name_convert

RULE_SET = []


class Rule:
    def __init__(self, name, func=None):
        """
        :param name: A rule name.
        :param func: The function that the rule logic applied. It accepts one parameter,
        :param sql_dict: Parsed SQL dict with following format,
            # {
            #    'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
            #    'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
            #    'where': condition
            #    'groupBy': [col_unit1, col_unit2, ...]
            #    'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
            #    'having': condition
            #    'limit': None/limit value
            #    'intersect': None/sql
            #    'except': None/sql
            #    'union': None/sql
            # }
        """
        self._name = name
        self._func = func

    def apply_rule(self, sql_dict):
        ret = self._func(sql_dict)
        return ret

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        else:
            return False


# Define the rules
# ### Rule 1 ###
def rule1_func(sql_dict):
    """
    This rule defines the query-scope syntax constraint when grouping exists.
    If grouping exists, there should have at least one aggregate in either SELECT or HAVING or ORDER BY.
    """
    if not sql_dict['groupBy']:
        return True

    has_agg = False
    # Check if aggregate exists in projection
    for sel in sql_dict['select'][1]:
        agg_id, _ = sel
        if agg_id > 0:
            has_agg = True
            break
    # Check if having exists
    if not has_agg and sql_dict['having']:
        has_agg = True
    # Check if aggregate exists in output
    if not has_agg and sql_dict['orderBy']:
        for val_unit in sql_dict['orderBy'][1]:
            _, col_unit, _ = val_unit
            agg_id, _, _ = col_unit
            if agg_id > 0:
                has_agg = True
                break

    return has_agg


# ### Rule 2 ###
def rule2_func(sql_dict):
    """
    This rule defines the orderby-clause-scope syntax constraint. 
    If grouping not exists, aggregate should not occur in ordering.
    """
    if not sql_dict['groupBy'] and sql_dict['orderBy']:
        # Check if aggregate exists in output
        for val_unit in sql_dict['orderBy'][1]:
            _, col_unit, _ = val_unit
            agg_id, _, _ = col_unit
            if agg_id > 0:
                return False

    return True


# ### Rule 3 ###
def rule3_func(sql_dict):
    """
    This rule defines the select-clause-scope syntax constraint with attribute-aggregate-mixing case. 
    If grouping not exists, there are 3 scenarios to be considered:
        1. Only aggregate functions;
        2. Only columns;
        3. Only star.
    """
    case_1 = True
    case_2 = True
    case_3 = True
    if not sql_dict['groupBy']:
        for sel in sql_dict['select'][1]:
            agg_id, val_unit = sel
            if agg_id == 0:
                case_1 = False
            if agg_id > 0:
                case_2 = False
            _, col_unit, _ = val_unit
            _, col, _ = col_unit
            if col != "__all__":
                case_3 = False

    return case_1 | case_2 | case_3


# ### Rule 4 ###
def rule4_func(sql_dict):
    """
    This rule defines several rules to limit the complexity that the SQL can be.
        There is only one BETWEEN...AND/LIKE-predicate exists.
    """
    between = 0
    like = 0
    if sql_dict['where']:
        for unit in sql_dict['where'][::2]:
            if unit[1] == 1:  # BETWEEN
                between += 1
            elif unit[1] == 9:  # LIKE
                like += 1

    if between > 1 or like > 1:
        return False

    return True


# ### RULE 5 ###
def rule5_func(sql_dict):
    """
    This rule defines for the group operator will compress the tuples, which will limit the projection ability.
    If the groupby exists, can not project non-primary col except for:
        1) group on primary col
        2) group on join condition
        3) group on this non-primary col
    Note that the rule is insufficient and necessary condition
    @param sql_dict: sql which is rebuilt
           primary_cols: columns which is 1-1 correspondence, which we think name, id, code, number, abbreviation so on
    @return: True or False
    """
    if not sql_dict['groupBy'] or len(sql_dict['groupBy']) > 1:
        return True

    group_by_col = sql_dict['groupBy'][0][1].strip('_')

    # 1) group on primary col
    primary_cols = sql_dict.get('primary_cols', [])
    if not primary_cols or group_by_col in primary_cols:
        return True

    # 2) group on join condition
    join_cols = list()
    for cond in sql_dict['from']['conds']:
        if isinstance(cond, tuple):
            join_cols.append(cond[2][1][1].strip('_'))
            join_cols.append(cond[3][1].strip('_'))
    if group_by_col in join_cols:
        return True

    # group on this non-primary col
    for select_unit in sql_dict['select'][1]:
        if select_unit[0] == 0:
            if select_unit[1][1][1].strip('_') != group_by_col:
                return False
    return True


# ### RULE 6 ###
def rule6_func(sql_dict: Dict):
    """
    Equal predicate is very strict, some sql should be removed because no logical result
    Situations will be removed:
        1) one col only can have one equal predicate if the predicate is concat by `and`
        2) project the equal predicate col
    @param sql_dict:
    @return: True or False
    """
    if not sql_dict['where']:
        return True

    # stat col num in where, which may lead to conflict (concat by `and`)
    equal_predicate_col_num = defaultdict(int)
    pre_logic = 'and'

    for predicate in sql_dict['where']:
        if isinstance(predicate, tuple) and predicate[1] == 2 and pre_logic == 'and':
            equal_predicate_col_num[predicate[2][1][1].strip('_')] += 1
        else:
            pre_logic = predicate

    # stat table num
    table_num = defaultdict(int)
    for _, table in sql_dict['from']['table_units']:
        table_num[table.strip('_')] += 1

    # 1) one col only can have one equal predicate if the predicate is concat by `and`
    for col in equal_predicate_col_num:
        if equal_predicate_col_num[col] > 1:
            if equal_predicate_col_num[col] > table_num[col.split('.')[0]]:
                return False

    # 2) only project the equal predicate col
    if len(sql_dict['select'][1]) == 1 and len(sql_dict['where']) == 1 and sql_dict['where'][0][1] == 2 \
            and sql_dict['select'][1][0][1][1][1] == sql_dict['where'][0][2][1][1] \
            and table_num[sql_dict['select'][1][0][1][1][1].strip('_').split('.')[0]] == 1:  # only one this col exists
        return False

    return True


# ### Rule 7 ###
def rule7_func(sql_dict):
    """
    This rule defines several rules to limit the complexity that the SQL can be.
        If nested query exists, no IUE/grouping/output;
    """
    has_nested = False
    if sql_dict['where']:
        for unit in sql_dict['where'][::2]:
            if isinstance(unit[3], dict):
                has_nested = True

    if has_nested and any(elem for elem in [sql_dict['intersect'], sql_dict['union'],
                                            sql_dict['except'], sql_dict['orderBy']]):
        return False

    return True


rule1 = Rule(name="group_aggregate_rule", func=rule1_func)
rule2 = Rule(name="order_aggregate_rule", func=rule2_func)
rule3 = Rule(name="project_mix_rule", func=rule3_func)
rule4 = Rule(name="between_like_rule", func=rule4_func)
rule5 = Rule(name="group_project_rule", func=rule5_func)
rule6 = Rule(name="equal_predicate_rule", func=rule6_func)
rule7 = Rule(name="nested_iue_rule", func=rule6_func)
RULE_SET.append(rule1)
RULE_SET.append(rule2)
RULE_SET.append(rule3)
RULE_SET.append(rule4)
RULE_SET.append(rule5)
RULE_SET.append(rule6)
RULE_SET.append(rule7)


def test():
    if 'nested_iue_rule' in RULE_SET:
        RULE_SET.remove('nested_iue_rule')
        print('\n')
        print('nested_iue_rule' in RULE_SET)
