# -*- coding: utf-8 -*-
# @Time    : 2021/6/4 13:37
# @Author  : 
# @Email   : 
# @File    : sql_tmp_update.py
# @Software: PyCharm
from datagen.sqlgen.qunit.unit_extract import SQLClauseSeparator, Source


def sql_nested_query_tmp_name_convert(sql: str, nested_level=0, sub_query_token='S') -> str:
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    tokens = sql.split()
    select_count = sql.lower().split().count('select')
    level_flag = sub_query_token * nested_level

    # recursive exit
    if select_count == 1:
        # need to fix the last level's tmp name
        res = sql
        if nested_level:
            # log all tmp name
            tmp_name_list = set()
            for i in range(len(tokens)):
                # find tmp name
                if tokens[i].lower() == 'as':
                    tmp_name_list.add(tokens[i + 1])
                # convert every tmp name
            for tmp_name in tmp_name_list:
                res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')
        return res

    # for new sql's token
    new_tokens = list()
    bracket_num = 0
    i = 0
    # iter every token in tokens
    while i < len(tokens):
        # append ordinary token
        new_tokens.append(tokens[i])
        # find a nested query
        if tokens[i] == '(' and tokens[i + 1].lower() == 'select':
            nested_query = ''
            bracket_num += 1
            left_bracket_position = i + 1
            # in one nested query
            while bracket_num:
                i += 1
                if tokens[i] == '(':
                    bracket_num += 1
                elif tokens[i] == ')':
                    bracket_num -= 1
                # to the end of the query
                if bracket_num == 0:
                    # format new nested query and get the tokens
                    nested_query = ' '.join(tokens[left_bracket_position: i])
                    nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 1)
            # new sql's token log
            new_tokens.append(nested_query)
            # append the right bracket
            new_tokens.append(tokens[i])
        # IUE handle
        elif tokens[i].lower() in {'intersect', 'union', 'except'}:
            nested_query = ' '.join(tokens[i + 1:])
            nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 10)
            new_tokens.append(nested_query)
            i += 9999
        i += 1
    # format the new query
    res = ' '.join(new_tokens)
    if nested_level:
        # log all tmp name
        tmp_name_list = set()
        for i in range(len(new_tokens)):
            # find tmp name
            if new_tokens[i].lower() == 'as':
                tmp_name_list.add(new_tokens[i + 1])
            # convert every tmp name
        for tmp_name in tmp_name_list:
            res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')

    return res


def use_alias(sql: str):
    """
    replace original relation name with alias to make sure that the sql unit align
    @param sql: original sql which may use the relation name to specify the col
    @return: new sql which use the alias
    """
    sql = sql.upper()
    # separate the original sql into clause
    sql_clause_separator = SQLClauseSeparator(sql)
    # get the source part for the alias mapping
    source = Source(sql_clause_separator.from_clause)
    # get invert index for replace
    invert_index = dict()
    for alias, relation in source.alias_mapping.items():
        invert_index[relation] = alias
    # replace every relation with alias
    for relation in invert_index:
        sql = sql.replace(f' {relation}.', f' {invert_index[relation]}.')

    sql = sql.replace(' __SingleTable__.', ' ')
    return sql


def test():
    test_sql = 'SELECT T2.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'T2.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE T2.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY T2.song_release_year ' \
               'ORDER BY COUNT ( T2.name ) LIMIT 5 INTERSECT ' \
               'SELECT T2.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'T2.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE T2.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY T2.song_release_year ' \
               'ORDER BY COUNT ( T2.name ) LIMIT 5'

    print(sql_nested_query_tmp_name_convert(test_sql))


def test2():
    test_sql = 'SELECT singer_in_concert.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'singer.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE singer.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY singer.song_release_year ' \
               'ORDER BY COUNT ( singer.name ) LIMIT 5 INTERSECT ' \
               'SELECT T2.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'T2.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE T2.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY T2.song_release_year ' \
               'ORDER BY COUNT ( T2.name ) LIMIT 5'

    gold_sql = 'SELECT T1.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'T2.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE T2.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY T2.song_release_year ' \
               'ORDER BY COUNT ( T2.name ) LIMIT 5 INTERSECT ' \
               'SELECT T2.song_release_year , COUNT ( * ) FROM singer_in_concert' \
               ' AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id WHERE ' \
               'T2.age > ( ' \
               'SELECT AVG ( age ) FROM T1.singer AS T1 JOIN singer_in_concert AS T2 ' \
               'WHERE T2.concert_id = ( SELECT MAX ( T1.ID ) FROM singer AS T1 ) ' \
               ')' \
               ' GROUP BY T2.song_release_year ' \
               'ORDER BY COUNT ( T2.name ) LIMIT 5'
    assert use_alias(test_sql) == gold_sql.upper()
