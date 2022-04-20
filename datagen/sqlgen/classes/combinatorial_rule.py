# 需要注意空格的使用
# terminal后面考虑空格问题
# non terminal考虑内部空格问题，不考虑开头结尾
COMBINATORIAL_RULE_DICTIONARY = {}

COMBINATORIAL_RULE_DICTIONARY['skeleton'] = \
    [
        ['<source>', '"|"', '<projection>'],
        ['<source>', '"|"', '<projection>', '"|"', '<selection>'],
        ['<source>', '"|"', '<grouping>', '"|"', '<projection>'],
        ['<source>', '"|"', '<projection>', '"|"', '<output>'],
        ['<iue>', '"|"', '<source>', '"|"', '<projection>'],

        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<selection>'],
        ['<source>', '"|"', '<projection>', '"|"', '<selection>', '"|"', '<output>'],
        ['<iue>', '"|"', '<source>', '"|"', '<projection>', '"|"', '<selection>'],
        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<output>'],
        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<iue>'],
        ['<source>', '"|"', '<projection>', '"|"', '<output>', '"|"', '<iue>'],

        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<selection>', '"|"', '<output>'],
        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<selection>', '"|"', '<iue>'],
        ['<source>', '"|"', '<projection>', '"|"', '<selection>', '"|"', '<output>', '"|"', '<iue>'],
        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<output>', '"|"', '<iue>'],

        ['<source>', '"|"', '<grouping>', '"|"', '<projection>', '"|"', '<selection>', '"|"', '<output>',
         '"|"', '<iue>']
    ]
COMBINATORIAL_RULE_DICTIONARY['<projection>'] = \
    [
        ['"SELECT "', 'attr_selects']
    ]
COMBINATORIAL_RULE_DICTIONARY['attr_selects'] = \
    [
        ['attr_select', '" , "', 'attr_selects'],
        ['attr_select'],
    ]

COMBINATORIAL_RULE_DICTIONARY['<source>'] = \
    [
        ['"FROM "', 'source']
    ]

COMBINATORIAL_RULE_DICTIONARY['<selection>'] = \
    [
        ['"WHERE "', 'predicate', '" "', 'where_conj'],
        ['"WHERE "', 'predicate']
    ]
COMBINATORIAL_RULE_DICTIONARY['predicate'] = \
    [
        ['pred_where'], # duplicate to make more possibilities
        ['pred_where'],
        ['pred_where'],
        ['pred_where_subquery']
    ]
COMBINATORIAL_RULE_DICTIONARY['where_conj'] = \
    [
        ['"AND "', 'predicate', '" "', 'where_conj'],
        ['"AND "', 'predicate'],
        ['"OR "', 'predicate', '" "', 'where_conj'],
        ['"OR "', 'predicate']
    ]

COMBINATORIAL_RULE_DICTIONARY['<grouping>'] = \
    [
        ['"GROUP BY "', 'attr_groups', '" HAVING "', 'pred_having', '" "', 'having_conj'],
        ['"GROUP BY "', 'attr_groups', '" HAVING "', 'pred_having'],
        ['"GROUP BY "', 'attr_groups']
    ]
COMBINATORIAL_RULE_DICTIONARY['having_conj'] = \
    [
        ['"AND "', 'pred_having', '" "', 'having_conj'],
        ['"AND "', 'pred_having']
    ]

COMBINATORIAL_RULE_DICTIONARY['attr_groups'] = \
    [
        ['attr_group', '" , "', 'attr_groups'],
        ['attr_group'],
    ]

COMBINATORIAL_RULE_DICTIONARY['<output>'] = \
    [
        ['"ORDER BY "', 'attr_orders'],
        ['argmax'],
        ['argmin']
    ]
COMBINATORIAL_RULE_DICTIONARY['attr_orders'] = \
    [
        ['attr_order', '" , "', 'attr_orders'],
        ['attr_order'],
    ]

COMBINATORIAL_RULE_DICTIONARY['<iue>'] = \
    [
        ['iues']
    ]
COMBINATORIAL_RULE_DICTIONARY['iues'] = \
    [
        ['iue', '" "', 'iues'],
        ['iue']
    ]

# 需要初始化
COMBINATORIAL_RULE_DICTIONARY['attr_select'] = []
COMBINATORIAL_RULE_DICTIONARY['source'] = []
COMBINATORIAL_RULE_DICTIONARY['pred_where'] = []
COMBINATORIAL_RULE_DICTIONARY['pred_where_subquery'] = []
COMBINATORIAL_RULE_DICTIONARY['attr_group'] = []
COMBINATORIAL_RULE_DICTIONARY['pred_having'] = []
COMBINATORIAL_RULE_DICTIONARY['attr_order'] = []
COMBINATORIAL_RULE_DICTIONARY['argmax'] = []
COMBINATORIAL_RULE_DICTIONARY['argmin'] = []
COMBINATORIAL_RULE_DICTIONARY['iue'] = []
