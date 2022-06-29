# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 14:38
# @Author  : 
# @Email   : 
# @File    : unit_bound_estimation.py
# @Software: PyCharm
from copy import deepcopy
from configs.config import DIR_PATH, QUNITS_FILE
import json
from collections import defaultdict
from scipy.special import comb


def accumulated_comb(base_num: int, max_choice_num: int) -> int:
    res = 0
    for i in range(1, max_choice_num + 1):
        res += comb(base_num, i)
    return res


def upper_bound_estimation(db_units: dict) -> int:
    # classify with unit cat
    cat_units = defaultdict(list)
    for unit in db_units['units']:
        cat_units[unit[1]].append(unit)
    # ---------------------------------- FOR DEBUG ----------------------------------
    print(cat_units.keys())
    # ********************************** END DEBUG **********************************
    # syntactic info
    syntactic = deepcopy(db_units['global_syntactic'])

    bound = 1
    bound *= len(cat_units['source'])

    bound *= accumulated_comb(len(cat_units['attr_select']), syntactic['max_projection_num'])
    bound *= accumulated_comb(len(cat_units['pred_where']) + len(cat_units['pred_where_subquery']),
                              syntactic['max_where_predicate_num'])
    return int(bound)


def main():
    res_list = list()
    # read in unit file
    with open(DIR_PATH + QUNITS_FILE, 'r') as f:
        units = json.load(f)
    for db_id in units:
        if db_id == 'concert_singer':
            print(db_id)
        upper_bound = upper_bound_estimation(units[db_id])
        res_list.append(upper_bound)
        print(f"[INFO] SQL num upper bound {upper_bound} @{db_id}")
    print(sorted(res_list, reverse=True))


if __name__ == '__main__':
    main()
