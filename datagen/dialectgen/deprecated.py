###
# Deprecated APIs
###
def convert_sql_to_desc(query_tokens: List[str], schema: Dict, _schema: Dict) -> str:
    q_dict = get_sql(_schema, query_tokens)
    
    sql_cls_descs = get_sql_clause_descs(q_dict, schema)
    sql_desc_parts = [c_desc for c_desc in sql_cls_descs if c_desc]
    sql_desc = ' '.join(sql_desc_parts)

    keyword = None
    if q_dict['intersect']: keyword = 'intersect'
    elif q_dict['except']: keyword = 'except'
    elif q_dict['union']: keyword = 'union'

    if keyword:
        sql_desc = ' '.join([sql_desc, WORD_DICT[keyword]])
        sql_cls_descs = get_sql_clause_descs(q_dict[keyword], schema, iue=True)
        subsql_desc_parts = [cls_desc for cls_desc in sql_cls_descs[2:] if cls_desc]
        subsql_desc = ' '.join(subsql_desc_parts)
        sql_desc = ' '.join([sql_desc, subsql_desc])

    # print(f"sql_desc:{sql_desc}")

    return sql_desc+WORD_DICT["?"]
    
def parse_val_unit(val_unit: tuple, shared_key: bool) -> str:
    unit_op, col_unit1, col_unit2 = val_unit
    agg_id1, col_id1, _ = col_unit1

    if col_id1 == "__all__": 
        col1 = WORD_DICT["all"] if agg_id1 == 0 else ""
    else: col1 = col_id1.split('.')[-1].strip('_').replace('_', ' ')

    tab1 = "" if (col_id1 == "__all__" or shared_key) \
        else col_id1.split('.')[0].strip('_').replace('_', ' ')
    
    col_unit1_t = ' '.join(
        [WORD_DICT[AGG_OPS[agg_id1]], col1, WORD_DICT["of"], tab1]) if tab1 \
            else ' '.join([WORD_DICT[AGG_OPS[agg_id1]], col1])

    if col_unit2:
        agg_id2, col_id2, _ = col_unit2

        if col_id2 == "__all__": 
            col2 = WORD_DICT["all"] if agg_id2 == 0 else ""
        else: col2 = col_id2.split('.')[-1].strip('_').replace('_', ' ')

        tab2 = "" if (col_id2 == "__all__" or shared_key) \
            else col_id2.split('.')[0].strip('_').replace('_', ' ')
        
        col_unit2_t = ' '.join(
            [WORD_DICT["the"], WORD_DICT[AGG_OPS[agg_id2]], col2, WORD_DICT["of"], tab2]) if tab2 \
                else ' '.join([WORD_DICT["the"], WORD_DICT[AGG_OPS[agg_id2]], col2])
        
        val_t = ' '.join([col_unit1_t, WORD_DICT[UNIT_OPS[unit_op]], col_unit2_t])
    else:
        val_t = col_unit1_t
        
    return ' '.join(val_t.split())

def parse_condition(cond: tuple, schema: Dict, shared_key: bool) -> str:
    not_op, op_id, val_unit, val1, val2 = cond
    val_t = parse_val_unit(val_unit, shared_key)
    
    not_t = WORD_DICT["not"] if not_op else ""
    if not isinstance(val1, dict): 
        cond_t = ' '.join([val_t, not_t, WORD_DICT[WHERE_OPS[op_id]], str(val1)])
        # "between and" case
        if op_id == 1 and val2: cond_t = ' '.join([cond_t, WORD_DICT["and"], str(val2)])
    else:
        cond_t = ' '.join([val_t, not_t, WORD_DICT[WHERE_OPS[op_id]]])
        clauses_t = get_sql_clause_descs(val1, schema=schema)
        for ii, t in enumerate(clauses_t[1:-1]):
            if t: cond_t += t
    
    return cond_t

def get_sql_clause_descs(q_dict: Dict, schema: Dict, iue: bool = False) -> List[str]:
    ####################################################################
    # from clause
    ####################################################################
    primary_keys = {}
    subject = ""
    from_cls_desc = ""
    tables = q_dict['from']['table_units']
    for table in tables:
        table_name = table[1].strip('_')
        # Extract the primary key for each relation as the subject candidate for the query
        key = None
        for column in schema[table_name].columns:
            if column.is_primary_key: 
                key = column.name.strip("id")
        if key and len(key) > 0: primary_keys[table_name] = key

    if len(tables) == 1:
        ### Only support table type
        table_name = tables[0][1].strip('_').replace('_', ' ')
        from_cls_desc = ' '.join([WORD_DICT['of'], table_name])
    else:
        # If the different relations are related to the same subject
        shared_key = len(list(set(list(primary_keys.values())))) == 1
        if shared_key: subject = list(primary_keys.values())[0].strip('id')
        if subject:
            subject = subject.strip('_').replace('_', ' ')
            from_cls_desc = ' '.join([WORD_DICT['of'], subject])

    shared_key = len(list(set(list(primary_keys.values())))) == 1 or len(tables) == 1

    from_cls_desc = ' '.join(from_cls_desc.split())
    # if from_cls_desc: print(f"from_cls_desc:{from_cls_desc}")
    ####################################################################
    # where clause
    # Format: (col_unit1 | unit_op | col_unit2) | not_op | op_id | val1
    ####################################################################
    where_cls_desc = ""
    if q_dict['where']:
        where_cls_desc = WORD_DICT['where']
    conds = q_dict['where'][::2]
    conjs = q_dict['where'][1::2]
    for i, cond in enumerate(conds):
        cond_t = parse_condition(cond, schema, shared_key)
        where_cls_desc = ' '.join([where_cls_desc, cond_t])
        if i < len(conjs): where_cls_desc = ' '.join([where_cls_desc, WORD_DICT[conjs[i]]])

    where_cls_desc = ' '.join(where_cls_desc.split())
    # if where_cls_desc: print(f"where_cls_desc:{where_cls_desc}")
    ####################################################################
    # select clause
    # agg_id (col_unit1 | unit_op | col_unit2)
    ####################################################################
    select_cls_desc = ""
    _, sel_cols = q_dict['select']
    for j, sel_col in enumerate(sel_cols):
        agg_id, val_unit = sel_col
        val_t = parse_val_unit(val_unit, shared_key)

        # special case: SELECT * FROM XXX
        if val_t:  sel_text = ' '.join([WORD_DICT["the"], WORD_DICT[AGG_OPS[agg_id]], val_t])
        else: sel_text = ' '.join([WORD_DICT[AGG_OPS[agg_id]], val_t])

        select_cls_desc = ' '.join([select_cls_desc, sel_text])
        if j+1 < len(sel_cols): select_cls_desc += ' '.join([select_cls_desc, WORD_DICT["and"]])
        elif not iue: select_cls_desc = ' '.join([WORD_DICT["what"], select_cls_desc])

    select_cls_desc = ' '.join(select_cls_desc.split())
    # if select_cls_desc: print(f"select_cls_desc:{select_cls_desc}")
    ####################################################################
    # groupBy clause
    ####################################################################
    groupby_cls_desc = ""
    if q_dict['groupBy']:
        groupby_cls_desc = WORD_DICT['groupby']
    col_units = q_dict['groupBy']
    for k, col_unit in enumerate(col_units):
        agg_id, col_id, _ = col_unit
        col1 = col_id.split('.')[-1].strip('_').replace('_', ' ') 
        tab1 = "" if shared_key else col_id.split('.')[0].strip('_').replace('_', ' ') 
        col_unit_t = ' '.join(
            [WORD_DICT[AGG_OPS[agg_id]], col1, WORD_DICT["of"], tab1]) if tab1 \
            else ' '.join([WORD_DICT[AGG_OPS[agg_id]], col1])
        
        groupby_cls_desc = ' '.join([groupby_cls_desc, col_unit_t])
        if k+1 < len(col_units): groupby_cls_desc = ' '.join([groupby_cls_desc, WORD_DICT["and"]])

    groupby_cls_desc = ' '.join(groupby_cls_desc.split())
    # if groupby_cls_desc: print(f"groupby_cls_desc:{groupby_cls_desc}")
    ####################################################################
    # having clause
    # Format: (col_unit1 | unit_op | col_unit2) | not_op | op_id | val1
    ####################################################################
    having_cls_desc = ""
    if q_dict["having"]:
        having_cls_desc = WORD_DICT['having']
    conds = q_dict['having'][::2]
    conjs = q_dict['having'][1::2]
    for l, cond in enumerate(conds):
        cond_t = parse_condition(cond, schema, shared_key)
        having_cls_desc = ' '.join([having_cls_desc, cond_t])
        if l < len(conjs): having_cls_desc = ' '.join([having_cls_desc, WORD_DICT[conjs[l]]])
    
    having_cls_desc = ' '.join(having_cls_desc.split())
    # if having_cls_desc: print(f"having_cls_desc:{having_cls_desc}")
    # if q_dict['having']: having_cls_desc += WORD_DICT['having']
    ####################################################################
    # orderby+limit clause
    ####################################################################
    orderby_cls_desc = ""
    if q_dict['orderBy']:
        d_asc, val_units = q_dict['orderBy']
        limit_n = q_dict['limit']

        if not limit_n: orderby_cls_desc = WORD_DICT["orderby"]

        for m, val_unit in enumerate(val_units):
            unit_op, col_unit1, col_unit2 = val_unit
            agg_id1, col_id1, _ = col_unit1

            col1 = "" if (col_id1 == "__all__") \
                else col_id1.split('.')[-1].strip('_').replace('_', ' ')
            col_unit1_t = ' '.join([WORD_DICT[AGG_OPS[agg_id1]], col1])

            if col_unit2:
                agg_id2, col_id2, _ = col_unit2
                col2 = "" if col_id2 == "__all__" \
                    else col_id2.split('.')[-1].strip('_').replace('_', ' ')

                col_unit2_t = ' '.join([WORD_DICT[AGG_OPS[agg_id2]], col2])
                val_t = ' '.join([col_unit1_t, WORD_DICT[UNIT_OPS[unit_op]],  col_unit2_t])
            else: val_t = col_unit1_t
            
            orderby_cls_desc = ' '.join([orderby_cls_desc, val_t])
            if m + 1 < len(val_units): orderby_cls_desc = ' '.join([orderby_cls_desc, WORD_DICT["and"]])

        if limit_n:
            if limit_n > 1:
                key = d_asc + "_limit_n" 
                orderby_cls_desc = ' '.join([WORD_DICT["orderby_limit"], WORD_DICT[key], str(limit_n), orderby_cls_desc])
            else:
                key = d_asc + "_limit_1"
                orderby_cls_desc = ' '.join([WORD_DICT["orderby_limit"], WORD_DICT[key], orderby_cls_desc])
    
    orderby_cls_desc = ' '.join(orderby_cls_desc.split())
    # if orderby_cls_desc: print(f"orderby_cls_desc:{orderby_cls_desc}")

    return [select_cls_desc, from_cls_desc, where_cls_desc, groupby_cls_desc, having_cls_desc, orderby_cls_desc]
