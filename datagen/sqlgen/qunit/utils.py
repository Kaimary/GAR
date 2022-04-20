# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 16:39
# @Author  : 
# @Email   : 
# @File    : utils.py
# @Software: PyCharm
import json
from collections import defaultdict
from typing import List
from .keywords import *
from spider_utils.utils import DBSchema


def alias_transformer(expression: str, alias_mapping: dict):
    """
    transformer one expression with alias to original relational name
    @param expression: expression which need to be replaced
    @param alias_mapping: alias mapping dict
    @return: expression with original name
    """
    for alias in alias_mapping:
        expression = expression.replace(f'{alias}.', f'{alias_mapping[alias]}.')
    return expression


def alias_remove(expression: str, alias_mapping: dict):
    """
    remove alias
    @param expression: expression which need to be removed
    @param alias_mapping: alias mapping dict
    @return: expression without alias
    """
    expression = ' ' + expression
    for alias in alias_mapping:
        expression = expression.replace(f' {alias}.', f' ')

    # format expression
    tokens = expression.split()
    # remove bracket in DISTINCT
    if 'DISTINCT' in tokens:
        index = tokens.index("DISTINCT")
        if tokens[index + 1] == '(':  # distinct has brackets
            bracket_num = 1
            i = index + 1
            while bracket_num and i < len(tokens):  # in bracket
                i += 1
                if tokens[i] == '(':
                    bracket_num += 1
                elif tokens[i] == ')':
                    bracket_num -= 1
            tokens.pop(i)  # pop `)`
            tokens.pop(index + 1)  # pop `(`
            expression = ' '.join(tokens)
    return expression.strip()


def alias_dependency_get(expression: str, alias_mapping: dict, db_schema: DBSchema):
    """
    get dependency by alias
    @param db_schema: db information for the SQL which ignore the alias
    @param expression: expression which need to be removed
    @param alias_mapping: alias mapping dict
    @return: expression without alias
    """
    # only one relation in the sql
    if len(alias_mapping) == 1:
        return list(alias_mapping.values())
    # more than one relation
    res = list()
    for alias in alias_mapping:
        if f' {alias}.' in ' ' + expression:
            res.append(alias_mapping[alias])

    # don't handle `*` in this function
    if '*' in expression:
        return res

    # ignore alias
    tokens = expression.split()
    attribute_expression = ''
    # get all attribute name from expression
    for token in tokens:
        if token in KEYWORDS:
            continue
        else:
            attribute_expression = token.lower()
    if db_schema and not res:  # have schema and alias analyse failed, start schema analyse
        for relation in alias_mapping.values():  # iter relations
            if attribute_expression in db_schema.table_column[relation.lower()]:  # get dep
                res.append(relation)
    return res


def get_all_sql(file_path='sqlgenv2/datasets/spider/train_dev_spider.json') -> dict:
    """
    get all sql dict
    @param file_path: sql file path
    @return: all sql dict with id key
    """
    res = defaultdict(list)
    with open(file_path, 'r') as f:
        spider_sqls = json.load(f)
    for spider_sql in spider_sqls:
        res[spider_sql['db_id']].append(spider_sql['query_toks_no_value'])
    return res


def get_all_sql_by_id(db_id: str, sql_file='sqlgenv2/datasets/spider/train_dev_spider.json') -> List[str]:
    """
    get sql list by db_id
    @param sql_file: query samples which need to be loaded
    @param db_id: spider id which want to be analysed
    @return: List of sql
    """
    sqls = get_all_sql(sql_file)[db_id]
    # fix sqls
    for i in range(len(sqls)):
        sqls[i] = ' '.join(sqls[i])
        # fix separated `.`
        sqls[i] = sqls[i].replace(' . ', '.')
        # fix separated operator
        for special_token in ['> =', '< =', '! =', '< >']:
            if special_token in sqls[i]:
                sqls[i] = sqls[i].replace(special_token, special_token[0] + special_token[-1])
        # make all tokens upper
        sqls[i] = sqls[i].upper()
    return sqls
