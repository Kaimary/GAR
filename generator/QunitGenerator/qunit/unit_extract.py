# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 14:30
# @Author  : 
# @Email   : 
# @File    : unit_extract.py
# @Software: PyCharm

"""
school:
select T1.name, T2.name from student as T1 join teacher as T2 on T1.mentor_id = T2.id where
    T1.age < (select avg(age) from student) order by T1.gpa desc limit 1
    union select T1.name, T2.name from student as T1 join teacher as T2 on T1.mentor_id = T2.id where
    T1.age < (select avg(age) from student) order by T1.gpa limit 1;
# get the top1 student's name and his mentor's name and the last1 student's name and his mentor's name under average age

Pattern demo
{
    'school': [
        ('student as T1 join teacher as T2 on T1.mentor_id = T2.id', 'source', [], 1),
        ('name', 'attr_select', ['student'], 1),
        ('name', 'attr_select', ['teacher'], 1),
        ('age _OP_ (select avg(age) from student)', 'pred_where', ['student'], 1),
        ('order by gpa desc limit 1', 'argmax', ['student'], 1)
        ('union select T1.name, T2.name from student as T1 join teacher as T2 on T1.mentor_id = T2.id where T1.age
            < (select avg(age) from student) order by T1.gpa limit 1', 'iue', ['student.name', 'teacher.name'], 1)
    ],
    'concert_singer': [
        ...
    ]
}


"""

import json
import os
from copy import deepcopy
from tqdm import tqdm
from .keywords import *
from typing import List
from .utils import get_all_sql, get_all_sql_by_id, alias_dependency_get, alias_remove
from configs.config import DIR_PATH, QUNITS_FILE, QUNITS_SET_COVER_FILE
from utils.spider_utils import DBSchema, get_all_schema


class PatternHashObject(object):
    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return f'{self.__str__()} {type(self).__name__}@{id(self)}'


class SQLClauseSeparator(PatternHashObject):
    def __init__(self, sql: str):
        self.sql = sql.strip()
        # fill up `;`
        if self.sql[-1] != ';':
            self.sql += ' ;'
        self.key_tokens_sql = {
            'SELECT': '',
            'FROM': '',
            'WHERE': '',
            'HAVING': '',
            'GROUP': '',
            'ORDER': '',
            'LIMIT': '',
            'IUE': '',
            ';': '',
        }
        self.__base_analyse()
        self.key_tokens_sql.pop(';')

    @property
    def select_clause(self):
        return self.key_tokens_sql['SELECT'] if len(self.key_tokens_sql['SELECT']) > 6 else ''

    @property
    def from_clause(self):
        return self.key_tokens_sql['FROM'] if len(self.key_tokens_sql['FROM']) > 4 else ''

    @property
    def where_clause(self):
        return self.key_tokens_sql['WHERE'] if len(self.key_tokens_sql['WHERE']) > 5 else ''

    @property
    def having_clause(self):
        return self.key_tokens_sql['HAVING'] if len(self.key_tokens_sql['HAVING']) > 6 else ''

    @property
    def group_clause(self):
        return self.key_tokens_sql['GROUP'] if len(self.key_tokens_sql['GROUP']) > 5 else ''

    @property
    def output_clause(self):
        res = self.key_tokens_sql['ORDER'] + ' ' + self.key_tokens_sql['LIMIT']
        return res if len(res) > 5 else ''

    @property
    def iue_clause(self):
        return self.key_tokens_sql['IUE'] if len(self.key_tokens_sql['IUE']) > 5 else ''

    def __base_analyse(self):
        """
        separate sql sub sentence into key tokens
        @return:  None
        """
        tokens = self.sql.split()
        pre_key_token = ''
        pre_key_token_position = 0
        bracket_num = 0
        # iter every token
        i = 0
        while i < len(tokens):
            # IUE clause detected
            if tokens[i] in IUE_KEYWORDS:
                self.key_tokens_sql[pre_key_token] = ' '.join(tokens[pre_key_token_position: i])
                self.key_tokens_sql['IUE'] = ' '.join(tokens[i:])
                break
            elif tokens[i] in self.key_tokens_sql:  # key token detect
                if not pre_key_token:  # fist key token
                    pre_key_token = tokens[i]
                    pre_key_token_position = i
                else:  # key token detected, sql sub sentence
                    self.key_tokens_sql[pre_key_token] = ' '.join(tokens[pre_key_token_position: i])
                    pre_key_token = tokens[i]
                    pre_key_token_position = i
            # skip brackets
            elif tokens[i] == '(':
                bracket_num += 1
                while bracket_num:
                    i += 1
                    if tokens[i] == ')':
                        bracket_num -= 1
                    elif tokens[i] == '(':
                        bracket_num += 1
            i += 1
        # check `;` at the end of clause
        for key in self.key_tokens_sql:
            if self.key_tokens_sql[key] and self.key_tokens_sql[key][-1] == ';':
                self.key_tokens_sql[key] = self.key_tokens_sql[key][:-2]
        return

    def __str__(self):
        res = {
            'select_clause': self.select_clause,
            'from_clause': self.from_clause,
            'where_clause': self.where_clause,
            'having_clause': self.having_clause,
            'group_clause': self.group_clause,
            'output_clause': self.output_clause,
            'iue_clause': self.iue_clause,
        }
        return json.dumps(res)


class BasicUnit(PatternHashObject):
    def __init__(self, expression='', catalog='', dependency=None, frequency=1):
        """
        Definition for a basic unit, as a quaternion
        @param expression: represent an atomic block of the semantics
            1) single attribute
            2) single predicate
            3) sql clause
            4) combination of multiple clauses
        @param catalog: syntactical type of the expression
            1) source
            2) attr_select | attr_group | attr_order
            3) pred_where | pred_having
            4) argmax | argmin
            5) iue
        @param dependency: semantic dependencies, defined as a list of expression it needed
        @param frequency: frequency of the query unit
        """
        if dependency is None:
            dependency = []
        self.expression = expression
        self.catalog = catalog
        self.dependency = deepcopy(dependency)
        self.frequency = frequency

    def str_without_frequency(self):
        """
        output the unit string without frequency
        @return: string of this basic unit
        """
        return f"('{self.expression}',{self.catalog},{self.dependency})"

    def __str__(self):
        return f"('{self.expression}',{self.catalog},{self.dependency},{self.frequency})"

    def __eq__(self, other):
        if isinstance(other, BasicUnit):
            return self.expression == other.expression and \
                   self.catalog == other.catalog and \
                   self.dependency == other.dependency
        else:
            raise TypeError("Basic Unit can not compare with other types")


class Units(PatternHashObject):
    def __init__(self):
        self.data: List[BasicUnit] = list()

    def append(self, basic_unit: BasicUnit):
        """
        add one basic unit into this units
        @param basic_unit: units which need to be added
        @return: None
        """
        # this basic unit has already in the data list
        if basic_unit in self.data:
            index = self.data.index(basic_unit)
            self.data[index].frequency += basic_unit.frequency
        # add a new basic unit to the data list
        else:
            self.data.append(deepcopy(basic_unit))

    def extend(self, other_units):
        """
        merge two units together
        @param other_units: other units instance
        @return: None
        """
        # merge two units
        if isinstance(other_units, Units):
            # add basic unit one by one
            for basic_unit in other_units.data:
                self.append(basic_unit)
        # can not merge with other types
        else:
            raise TypeError("can not extend a Units instance with other types")

    def str_without_frequency(self):
        """
        output the unit string without frequency
        @return: string of this units
        """
        return [f'({basic_unit.expression}, {basic_unit.catalog}, {basic_unit.dependency})' for basic_unit in self.data]

    def data_with_frequency(self):
        """
        output the unit string without frequency
        @return: string of this units
        """
        return [(f"'{basic_unit.expression}'", basic_unit.catalog, basic_unit.dependency, basic_unit.frequency)
                for basic_unit in self.data]

    def __str__(self):
        res = "["
        for basic_unit in sorted(self.data):
            res += f'{str(basic_unit)},'
        if res == "[":
            return ''
        res = res[:-1] + ']'
        return res

    def __add__(self, other):
        if isinstance(other, Units):
            units = Units()
            units.extend(self)
            units.extend(other)
            return units
        else:
            raise TypeError("can not add a Units instance with other types")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__str__()


class Clause(PatternHashObject):
    """
    Clause for every part of clause as a Father
    """

    def __init__(self, sql_clause: str = '',
                 alias_mapping: dict = None,
                 root_source: str = '',
                 schema: DBSchema = None):
        """
        init a clause obj
        @param sql_clause: sql clause
        @param alias_mapping: alias mapping dict
        @param root_source: root source dependency
        @param schema: db schema for dep analyse
        """
        self.schema = schema
        self.sql_clause = sql_clause.strip().upper()
        if alias_mapping:
            self.alias_mapping = deepcopy(alias_mapping)
        else:
            self.alias_mapping = dict()
        self.root_source = root_source.strip()
        self.units = Units()

        # for stat the predicate's num
        self.predicate_num = 0

    def attribute_unit_analyse(self, pure_clause: str, clause_name: str):
        """
        analyse attribute units, split by `,`
        @param pure_clause: clause without keywords
        @param clause_name: attribute category name
        @return: None
        """
        # split by `,`
        exps = pure_clause.split(',')
        # iter every expressions
        for raw_exp in exps:
            if not raw_exp.strip():  # empty
                continue
            # add new basic unit
            exp = alias_remove(raw_exp.strip(), self.alias_mapping)
            dep = alias_dependency_get(raw_exp.strip(), self.alias_mapping, self.schema)
            if '*' in exp or not dep:  # handle `*`
                dep = [self.root_source]
            self.units.append(BasicUnit(exp, f'{clause_name}', dep, 1))

    def predicate_unit_analyse(self, pure_clause: str, clause_name: str):
        """
        analyse predicate units, split by connection keywords
        @param pure_clause: clause without keywords
        @param clause_name: predicate category name
        @return: None
        """
        # split into tokens list
        tokens = pure_clause.split()
        tokens.append('OR')  # added as a EOS token

        # get predicates one by one
        last_index = 0
        bracket_num = 0
        subquery = ''
        operator = ''
        i = 0
        while i < len(tokens):
            if tokens[i] in PREDICATE_CMP_KEYWORDS:
                operator += tokens[i]
                if operator == 'NOT':
                    operator += ' '
                if tokens[i] == 'BETWEEN':  # skip between
                    while tokens[i] != 'AND':  # skip this `AND`
                        i += 1
                    i += 1
            elif tokens[i] == '(':  # skip bracket and judge if there has a subquery
                bracket_start_index = i
                bracket_num += 1
                while bracket_num:
                    i += 1
                    if tokens[i] == '(':
                        bracket_num += 1
                    elif tokens[i] == ')':
                        bracket_num -= 1
                    if bracket_num == 0:
                        break
                subquery = ' '.join(tokens[bracket_start_index: i + 1])
                if 'SELECT' not in subquery:  # subquery judgement
                    subquery = ''

            # find one connect keyword
            if tokens[i] in COND_OPS:
                final_index = last_index
                while tokens[final_index] not in PREDICATE_CMP_KEYWORDS:
                    final_index += 1
                attr_predicate = ' '.join(tokens[last_index: final_index])
                if subquery:
                    # has subquery
                    exp = f'{alias_remove(attr_predicate, self.alias_mapping)} {operator} {subquery}'
                    # ---------------------------------- FOR DEBUG HARDCODE ----------------------------------
                    # TODO: need delete later
                    # hard code to process iue in subquery
                    ignore_iue = False
                    for iue_keyword in IUE_KEYWORDS:
                        if iue_keyword in subquery:
                            ignore_iue = True
                    if not ignore_iue:
                        dep = alias_dependency_get(attr_predicate, self.alias_mapping, self.schema)
                        if '*' in exp or not dep:  # handle `*`
                            dep = [self.root_source]
                        self.units.append(BasicUnit(exp, f'pred_{clause_name}_subquery', dep, 1))
                    # ********************************** END DEBUG HARDCODE **********************************
                    # dep = alias_dependency_get(attr_predicate, self.alias_mapping, self.schema)
                    # if '*' in exp or not dep:  # handle `*`
                    #     dep = [self.root_source]
                    # self.units.append(BasicUnit(exp, f'pred_{clause_name}_subquery', dep, 1))
                else:
                    # do not has subquery
                    if 'BETWEEN' in operator:
                        exp = f'{alias_remove(attr_predicate, self.alias_mapping)} BETWEEN VALUE AND VALUE'
                    else:
                        exp = f'{alias_remove(attr_predicate, self.alias_mapping)} {operator} VALUE'
                    dep = alias_dependency_get(attr_predicate, self.alias_mapping, self.schema)
                    if '*' in exp or not dep:  # handle `*`
                        dep = [self.root_source]
                    self.units.append(BasicUnit(exp, f'pred_{clause_name}', dep, 1))
                subquery = ''
                operator = ''
                last_index = i + 1
                self.predicate_num += 1
            i += 1

    def get_subquery_num(self):
        """
        get subquery unit number
        @return: the num of subquery unit
        """
        res = 0
        for basic_unit in self.units.data:
            if 'subquery' in basic_unit.catalog:
                res += 1
        return res

    def __str__(self):
        return str(self.units)

    def __add__(self, other):
        """
        define the add action for Clause so that we can add every clause together
        @param other: the other one to be added
        @return: an Units object
        """
        if isinstance(other, Clause):
            res = Clause()
            res.units.extend(self.units)
            res.units.extend(other.units)
            return res
        else:
            raise TypeError("Clause only can added with other clause")


class Source(Clause):
    """
    Source part for one sql
    """

    def __init__(self, sql_clause: str):
        super(Source, self).__init__(sql_clause=sql_clause, alias_mapping={}, root_source='')
        # SQL clause check
        if not self.sql_clause.startswith('FROM'):
            raise Exception("Error sql clause in SOURCE part")

        self.__unit_analyse()

    def __unit_analyse(self):
        exp = self.sql_clause[5:]  # remove 'FROM'
        tokens = exp.split()

        if ' AS ' in exp:
            # get relation alias dict
            for i in range(len(tokens)):
                # get one alias, added into dict
                if tokens[i] == 'AS':
                    self.alias_mapping[tokens[i + 1]] = tokens[i - 1]
            try:
                exp = self.__reorder_join(exp)
            except ValueError:
                exp = exp
        elif ' JOIN ' in exp:
            # join but not has alias name, set the alias as it's original relation name
            self.alias_mapping[tokens[0]] = tokens[0]
            for i in range(1, len(tokens)):
                # find one join relation
                if tokens[i] == 'JOIN':
                    self.alias_mapping[tokens[i + 1]] = tokens[i + 1]
        else:
            # only one relation, add one default map
            self.alias_mapping['__SingleTable__'] = exp
        self.units.append(BasicUnit(exp, 'source', [], 1))

    def __reorder_join(self, exp: str):
        """
        reorder the join sequence
        @param exp: original join order
        @return: new exp which is ordered
        """
        # mark the original exp to check the res
        original_exp = exp
        # transfer the original alias to tmp alias for order
        for alias in self.alias_mapping:
            exp = exp.replace(f' {alias}.', f' {self.alias_mapping[alias]}.{alias}.')  # transfer ` alias.`
            exp = exp.replace(f' {alias} ', f' {self.alias_mapping[alias]}.{alias} ')  # transfer ` alias `

        tokens = exp.split()
        join_condition_list = list()
        i = 0
        while i < len(tokens):
            if tokens[i] == 'ON':  # find one join condition
                i += 1
                if tokens[i + 1] != '=':
                    raise ValueError('Not an equal join')
                # order one join condition
                if tokens[i] < tokens[i + 2]:
                    join_condition = (tokens[i], tokens[i + 2])
                else:
                    join_condition = (tokens[i + 2], tokens[i])
                join_condition_list.append(join_condition)
            i += 1
        # order all join condition
        join_condition_list.sort()

        def get_table_alias(table_alias_col):
            """
            get `table.alias` from `table.alias.col`
            @param table_alias_col: `table.alias.col`
            @return: `table.alias`
            """
            return table_alias_col[:table_alias_col.rindex('.')]

        def get_table(table_alias_col):
            """
            get `table` from `table.alias.col`
            @param table_alias_col: `table.alias.col`
            @return: `table`
            """
            return table_alias_col[:table_alias_col.index('.')]

        # no join condition, ignore with original exp
        if not join_condition_list:
            # print("Change error happened because NATURAL JOIN")
            return original_exp

        # traverse every join condition to reorder the join exp
        join_condition = join_condition_list.pop(0)
        new_alias_mapping = dict()
        # get initial element
        try:
            tables = [get_table_alias(join_condition[0]), get_table_alias(join_condition[1])]
        except ValueError:  # some buggy sql
            return original_exp
        new_alias_mapping[tables[0]] = f'T{len(new_alias_mapping) + 1}'
        new_alias_mapping[tables[1]] = f'T{len(new_alias_mapping) + 1}'
        # initial res
        res = f' {tables[0]} AS {new_alias_mapping[tables[0]]} JOIN {tables[1]} AS {new_alias_mapping[tables[1]]} ' \
              f'ON {join_condition[0]} = {join_condition[1]} '
        # traverse every join condition
        while join_condition_list:
            table_alias = ''
            # get one condition which can be added as next one
            for join_condition in join_condition_list:
                tables = [get_table_alias(join_condition[0]), get_table_alias(join_condition[1])]
                # select the new table to be added as join
                if tables[0] in new_alias_mapping:
                    table_alias = tables[1]
                    break
                if tables[1] in new_alias_mapping:
                    table_alias = tables[0]
                    break
            # the graph is not connected
            if not table_alias:
                raise ValueError("No table.alias found can be used for connect")
            # add new join condition
            new_alias_mapping[table_alias] = f'T{len(new_alias_mapping) + 1}'
            res += f'JOIN {table_alias} AS {new_alias_mapping[table_alias]} ' \
                   f'ON {join_condition[0]} = {join_condition[1]} '
            # removed the added join condition
            join_condition_list.remove(join_condition)
        # format the res
        for table_alias in new_alias_mapping:
            res = res.replace(f' {table_alias}.', f' {new_alias_mapping[table_alias]}.')
            res = res.replace(f' {table_alias} ', f' {get_table(table_alias)} ')
        res = res.strip()
        # make sure that the res is enough
        if len(res) != len(original_exp):
            # print("Change error happened after reorder the exp")
            return original_exp
        return res

    @property
    def get_root_source(self):
        """
        get root source
        @return: source expression
        """
        return self.units.data[0].expression


class GroupBy(Clause):
    """
    GroupBy part for one sql
    """

    def __init__(self, sql_clause: str, alias_mapping: dict, root_source: str = '', schema: DBSchema = None):
        super(GroupBy, self).__init__(sql_clause=sql_clause,
                                      alias_mapping=alias_mapping,
                                      root_source=root_source,
                                      schema=schema)
        if not self.sql_clause:
            return

        # SQL clause check
        if not self.sql_clause.startswith('GROUP BY'):
            raise Exception("Error sql clause in GROUP BY part")

        self.__unit_analyse()

    def __unit_analyse(self):
        pure_clause = self.sql_clause[9:]
        super(GroupBy, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='attr_group')


class Projection(Clause):
    """
    Projection part for one sql
    """

    def __init__(self, sql_clause: str, alias_mapping: dict, root_source: str = '', schema: DBSchema = None):
        super(Projection, self).__init__(sql_clause=sql_clause,
                                         alias_mapping=alias_mapping,
                                         root_source=root_source,
                                         schema=schema)
        # SQL clause check
        if not self.sql_clause.startswith('SELECT'):
            raise Exception("Error sql clause in PROJECTION part")

        self.__unit_analyse()

    def __unit_analyse(self):
        pure_clause = self.sql_clause[7:]
        super(Projection, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='attr_select')

    def asterisk_info_add(self, groupby: GroupBy):
        """
        add asterisk group by dependency
        @param groupby: group by col info as GroupBy
        @return: None
        """
        # No group by or no join, return
        if not len(groupby.units) or len(self.alias_mapping) <= 1:
            return
        # # more than one group by col, warning
        # if len(groupby.units) != 1:
        #     raise ValueError("[ERR] more than one group by col detected")

        # get group by col expression
        groupby_col = groupby.units.data[0].expression
        # add dependency for expression
        for i in range(len(self.units.data)):
            if '*' in self.units.data[i].expression and len(groupby.units) != 1:
                self.units.data[i].dependency.append(groupby_col)
                print('-' * 100)

    @property
    def get_as_dependency_list(self):
        """
        projection will be the dependency for iue part
        @return: projection list with its dependency
        """

        def reformat(basic_unit: BasicUnit):
            """
            reformat basic unit for iue dependency
            @param basic_unit: units in projection clause
            @return: list of formatted dependency
            """
            res = ''
            # find the dependency's relation position
            tokens = basic_unit.expression.split()
            for i in range(len(tokens)):
                if tokens[i] in KEYWORDS:  # not a column
                    i += 1
                else:
                    tokens.insert(i, '.')
                    tokens.insert(i, basic_unit.dependency[0])  # get the column which is need to be specified
                    break
            res += ' '.join(tokens)
            res = res.replace(' . ', '.')
            # handle DISTINCT
            if 'DISTINCT ' in basic_unit.expression:
                res = res.replace('DISTINCT ', '')
                res = 'DISTINCT ' + res
            return res

        return [reformat(basic_unit) for basic_unit in self.units.data]


class Where(Clause):
    """
    WhereConstrict part for one sql
    """

    def __init__(self, sql_clause: str, alias_mapping: dict, root_source: str = '', schema: DBSchema = None):
        super(Where, self).__init__(sql_clause=sql_clause,
                                    alias_mapping=alias_mapping,
                                    root_source=root_source,
                                    schema=schema)
        if not self.sql_clause:
            return

        # SQL clause check
        if not self.sql_clause.startswith('WHERE'):
            raise Exception("Error sql clause in WHERE part")

        self.__unit_analyse()

    def __unit_analyse(self):
        pure_clause = self.sql_clause[6:]
        super(Where, self).predicate_unit_analyse(pure_clause=pure_clause, clause_name='where')


class Having(Clause):
    """
    HavingConstrict part for one sql
    """

    def __init__(self, sql_clause: str, alias_mapping: dict, root_source: str = '', schema: DBSchema = None):
        super(Having, self).__init__(sql_clause=sql_clause,
                                     alias_mapping=alias_mapping,
                                     root_source=root_source,
                                     schema=schema)
        if not self.sql_clause:
            return

        # SQL clause check
        if not self.sql_clause.startswith('HAVING'):
            raise Exception("Error sql clause in HAVING part")

        self.__unit_analyse()

    def __unit_analyse(self):
        pure_clause = self.sql_clause[7:]
        super(Having, self).predicate_unit_analyse(pure_clause=pure_clause, clause_name='having')


class Output(Clause):
    """
    Output part for one sql
    """

    def __init__(self, sql_clause: str, alias_mapping: dict, root_source: str = '', schema: DBSchema = None):
        super(Output, self).__init__(sql_clause=sql_clause,
                                     alias_mapping=alias_mapping,
                                     root_source=root_source,
                                     schema=schema)
        if not self.sql_clause:
            return

        # SQL clause check
        if not self.sql_clause.startswith('ORDER BY'):
            raise Exception("Error sql clause in OUTPUT part")

        self.__unit_analyse()

    def __unit_analyse(self):
        # argmax | argmin
        if 'LIMIT ' in self.sql_clause:
            if ' DESC ' in self.sql_clause:
                pure_clause = self.sql_clause.split(' DESC ')[0][9:]
                super(Output, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='argmax')
            elif ' ASC ' in self.sql_clause:
                pure_clause = self.sql_clause.split(' ASC ')[0][9:]
                super(Output, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='argmin')
            else:
                pure_clause = self.sql_clause.split(' LIMIT ')[0][9:]
                super(Output, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='argmin')
        # only order semantic
        # elif 'LIMIT' in self.sql_clause:
        #     pure_clause = self.sql_clause.split(' LIMIT ')[0][9:]
        #     super(Output, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='attr_order')
        else:
            pure_clause = self.sql_clause[9:]
            super(Output, self).attribute_unit_analyse(pure_clause=pure_clause, clause_name='attr_order')


class IUE(Clause):
    """
        IUE part for one sql
    """

    def __init__(self, sql_clause: str, dependency_list: List[str], alias_mapping: dict, root_source: str = ''):
        super(IUE, self).__init__(sql_clause=sql_clause, alias_mapping=alias_mapping, root_source=root_source)
        self.dependency_list = dependency_list

        if not self.sql_clause:
            return

        # SQL clause check
        if not (self.sql_clause.startswith('INTERSECT') or self.sql_clause.startswith('UNION') or
                self.sql_clause.startswith('EXCEPT')):
            raise Exception("Error sql clause in IUE part")

        self.__unit_analyse()

    def __unit_analyse(self):
        exp = self.sql_clause
        cat = 'iue'
        dep = self.dependency_list
        self.units.append(BasicUnit(exp, cat, dep, 1))


class GlobalSyntactic(PatternHashObject):
    def __init__(self):
        self.max_projection_num = 0
        self.max_where_predicate_num = 0
        self.max_having_predicate_num = 0
        self.has_output = False
        self.has_iue = False
        self.has_group = False

        self.max_group_num = 0
        self.max_where_nested_num = 0
        self.max_having_nested_num = 0
        self.max_iue_num = 0
        self.max_order_num = 0
        self.has_where = False

    def dict(self):
        return {
            'max_projection_num': self.max_projection_num,
            'max_where_predicate_num': self.max_where_predicate_num,
            'max_having_predicate_num': self.max_having_predicate_num,
            'has_output': self.has_output,
            'has_iue': self.has_iue,
            'has_group': self.has_group,

            'max_group_num': self.max_group_num,
            'max_where_nested_num': self.max_where_nested_num,
            'max_having_nested_num': self.max_having_nested_num,
            'max_iue_num': self.max_iue_num,
            'max_order_num': self.max_order_num,
            'has_where': self.has_where,
        }

    def __add__(self, other):
        if isinstance(other, GlobalSyntactic):
            res = GlobalSyntactic()
            res.max_projection_num = max(self.max_projection_num, other.max_projection_num)
            res.max_where_predicate_num = max(self.max_where_predicate_num, other.max_where_predicate_num)
            res.max_having_predicate_num = max(self.max_having_predicate_num, other.max_having_predicate_num)
            res.has_output = max(self.has_output, other.has_output)
            res.has_iue = max(self.has_iue, other.has_iue)
            res.has_group = max(self.has_group, other.has_group)

            res.max_group_num = max(self.max_group_num, other.max_group_num)
            res.max_where_nested_num = max(self.max_where_nested_num, other.max_where_nested_num)
            res.max_having_nested_num = max(self.max_having_nested_num, other.max_having_nested_num)
            res.max_iue_num = max(self.max_iue_num, other.max_iue_num)
            res.max_order_num = max(self.max_order_num, other.max_order_num)
            res.has_where = max(self.has_where, other.has_where)
            return res
        else:
            raise TypeError("GlobalSyntactic only can added with other GlobalSyntactic")

    def __str__(self):
        return str(self.dict())


class SQLUnit(PatternHashObject):
    def __init__(self, sql: str, schema: DBSchema = None):
        self.schema = schema

        self.sql = sql.upper()  # upper all tokens in the sql
        if self.sql[-1] != ';':
            self.sql += ' ;'  # add `;` for separate
        self.separator = SQLClauseSeparator(self.sql)
        self.units = Units()
        self.global_syntactic = GlobalSyntactic()
        self.skeleton = ''

        self.source = Source(self.separator.from_clause)
        self.skeleton += '<source>'

        self.projection = Projection(self.separator.select_clause, self.source.alias_mapping,
                                     self.source.get_root_source, self.schema)
        self.global_syntactic.max_projection_num = len(self.projection.units)
        if self.global_syntactic.max_projection_num:
            self.skeleton += '|<projection>'

        self.groupby = GroupBy(self.separator.group_clause, self.source.alias_mapping,
                               self.source.get_root_source, self.schema)
        self.global_syntactic.has_group = len(self.groupby.units) > 0
        self.global_syntactic.max_group_num = len(self.groupby.units)
        if self.global_syntactic.has_group:
            self.skeleton += '|<grouping>'
            # self.projection.asterisk_info_add(self.groupby)

        self.where = Where(self.separator.where_clause, self.source.alias_mapping,
                           self.source.get_root_source, self.schema)
        self.global_syntactic.max_where_predicate_num = self.where.predicate_num
        self.global_syntactic.has_where = len(self.where.units) > 0
        self.global_syntactic.max_where_nested_num = self.where.get_subquery_num()
        if self.global_syntactic.max_where_predicate_num:
            self.skeleton += '|<selection>'

        self.having = Having(self.separator.having_clause, self.source.alias_mapping,
                             self.source.get_root_source, self.schema)
        self.global_syntactic.max_having_predicate_num = self.having.predicate_num
        self.global_syntactic.max_having_nested_num = self.having.get_subquery_num()

        self.output = Output(self.separator.output_clause, self.source.alias_mapping,
                             self.source.get_root_source, self.schema)
        self.global_syntactic.has_output = len(self.output.units) > 0
        self.global_syntactic.max_order_num = len(self.output.units)
        if self.global_syntactic.has_output:
            self.skeleton += '|<output>'

        self.iue = IUE(self.separator.iue_clause, self.projection.get_as_dependency_list,
                       self.source.alias_mapping, self.source.get_root_source)
        self.global_syntactic.has_iue = len(self.iue.units) > 0
        self.global_syntactic.max_iue_num = len(self.iue.units)
        if self.global_syntactic.has_iue:
            self.skeleton += '|<iue>'

        self.clauses = [self.source, self.projection, self.groupby, self.where, self.having, self.output, self.iue]
        self.__unit_gathering()

    def __unit_gathering(self):
        res = Clause()
        for clause in self.clauses:
            res += clause
        self.units = deepcopy(res.units)

    def __str__(self):
        return str(self.units)


class SpiderPattern(object):
    def __init__(self, sqls: List[str], schema: DBSchema = None):
        self.schema = schema

        self.units = Units()
        self.sql_unit_dict = dict()
        self.sqls: List[str] = sqls
        self.global_syntactic = GlobalSyntactic()
        self.skeleton = set()
        for sql in tqdm(self.sqls):
            sql_unit = SQLUnit(sql, schema)
            # add index unit to identified
            key = sql_unit.sql + f" @{len(self.sql_unit_dict)}"
            self.sql_unit_dict[key] = sql_unit.units.str_without_frequency()
            self.sql_unit_dict[key].append(sql_unit.skeleton)
            self.units += sql_unit.units
            self.global_syntactic += sql_unit.global_syntactic
            self.skeleton.add(sql_unit.skeleton)
        self.skeleton = sorted(list(self.skeleton), key=lambda x: len(x))

    def __str__(self):
        return str(self.units)


# @click.command()
# @click.argument("dataset", default='spider')
# @click.argument("schema_file", default="sqlgenv2/datasets/spider/tables.json",
#                 type=click.Path(exists=True, dir_okay=False))
# @click.argument("sql_file", default='sqlgenv2/datasets/spider/train_dev_spider.json',
#                 type=click.Path(exists=True, dir_okay=False))
def extract_spider_unit(dataset='spider', schema_file='datasets/spider/tables.json',
                        sql_file='datasets/spider/train_dev_spider.json',
                        db_path='datasets/spider/database/'):
    db_schemas = get_all_schema(schema_file)
    spider_patterns = dict()
    spider_set_cover = dict()
    count = 0
    all_sqls = get_all_sql(sql_file)
    for key in db_schemas.keys():
        count += 1
        print(f'[INFO] Handling DB ({count:>3}/{len(db_schemas)}): {key}\r')
        spider_pattern = SpiderPattern(get_all_sql_by_id(key, all_sqls), DBSchema(key, db_schemas, db_path))
        spider_patterns[key] = dict()
        spider_patterns[key]['global_syntactic'] = spider_pattern.global_syntactic.dict()
        spider_patterns[key]['units'] = spider_pattern.units.data_with_frequency()
        spider_patterns[key]['skeleton'] = spider_pattern.skeleton
        spider_set_cover[key] = spider_pattern.sql_unit_dict

    # check dir exists
    dir_path = (DIR_PATH + QUNITS_FILE.format(dataset)).rsplit('/', 1)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(DIR_PATH + QUNITS_FILE.format(dataset), 'w') as f:
        json.dump(spider_patterns, f, indent=4)
    with open(DIR_PATH + QUNITS_SET_COVER_FILE.format(dataset), 'w') as f:
        json.dump(spider_set_cover, f, indent=4)


if __name__ == '__main__':
    extract_spider_unit()
