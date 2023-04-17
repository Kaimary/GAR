from typing import Dict
import networkx as nx

from collections import defaultdict
from utils.evaluation.process_sql import WHERE_OPS, AGG_OPS
from configs.config import USE_ANNOTATION
"""
Generic Templates
"""
TEMPLATE = {
    "mu": "of",
    "theta": "that",
    "r": "",
    "distnt": "distinct",
    "h": "having",
    "gamma": "group by",
    "ao": "in ascending order of",
    "do": "in descending order of",
    "o": "in order of",
    "VAL_SEL": " that ",
    "COORD_CONJ": " and ",
    "CONJ_NOUN": " that ",
    "CONJ_PROJ": " and ",
    "CONJ_SEL": " and ",
    ">": "greater than",
    "year>": "after",
    "<": "less than",
    "year<": "before",
    "=": "is",
    "<=": "does not exceed",
    "year<=": "is or before",
    ">=": "does not less than",
    "year>=": "is or after",
    "!=": "does not equals to",
    "between": "is between",
    "like": "looks like",
    "not like": "not like",
    "in": "exists",
    "not in": "does not exist",
    "count": "number of",
    "max": "maximum",
    "min": "minimum",
    "avg": "average",
    "sum": "summary of",
    "join": "with the same",
    "limit": "top",
    "distinct": "different",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten"
}


def s_strip(string: str) -> str:
    if '##' in string:
        string = string.split('##')[0]
    if '.' in string:
        string = string.split('.')[1]

    return string


def build_graph_from_cond_unit(G, cond_unit, conj, table_dict, schema, node_posttag_dict, star_label, edge_type="mu", use_annotation=True):
    # TODO val2 processing
    not_op, op_id, val_unit, val1, val2 = cond_unit
    # TODO col_unit2 processing
    _, col_unit1, _ = val_unit
    col_name_node, is_year_col = build_graph_from_col_unit(G, col_unit1, table_dict, schema, node_posttag_dict, star_label,
                                                           edge_type=edge_type, conj=conj, use_annotation=use_annotation)

    op_opt = "not " + WHERE_OPS[op_id] if not_op else WHERE_OPS[op_id]
    op_opt_label = op_opt
    # If it is date-related column, we hard-code to add "year" prefix into the compare operator,
    # and use date-related templates to generate the label.
    if is_year_col and ('year'+op_opt) in TEMPLATE:
        op_opt_label = 'year'+op_opt
    if not isinstance(val1, Dict):
        # for value node, we use the string name as the node name, and store the value in attrs instead.
        val1_node = str(val1) + '##' + str(node_posttag_dict[str(val1)])
        G.add_node(val1_node, type="value_node", primary=False,
                   value=val1, label=str(val1).strip('"'))
        node_posttag_dict[str(val1)] += 1
        G.add_edge(
            col_name_node, val1_node, type=op_opt,
            label=table_dict['annotations'][op_opt_label] if use_annotation and 'annotations' in table_dict.keys(
            ) and op_opt_label in table_dict['annotations'].keys() else TEMPLATE[op_opt_label]
        )
        # BETWEEN AND case
        if val2:
            val2_node = str(val2) + '##' + str(node_posttag_dict[str(val2)])
            G.add_node(val2_node, type="value_node", primary=False,
                       value=val2, label=str(val2).strip('"'))
            node_posttag_dict[str(val2)] += 1
            G.add_edge(
                col_name_node, val2_node, type=op_opt,
                label=table_dict['annotations'][op_opt] if use_annotation and 'annotations' in table_dict.keys() and op_opt in table_dict['annotations'].keys() else TEMPLATE[
                    op_opt]
            )
    else:
        G1 = nx.MultiDiGraph(type="nested_node")
        G1, Rq = build_graph_from_sql(
            G1, val1, table_dict, schema, parent_op=op_opt, parent_node=col_name_node)
        G1.graph['Rq'] = Rq
        G.add_node(G1)
        G.add_edge(
            col_name_node, G1, type=op_opt,
            label=table_dict['annotations'][op_opt] if use_annotation and 'annotations' in table_dict.keys(
            ) and op_opt in table_dict['annotations'].keys() else TEMPLATE[op_opt]
        )

    return G


def build_graph_from_col_unit(G, col_unit1, table_dict, schema, node_posttag_dict, star_label, edge_type="mu",
                              reverse=False, sel_cols=None, conj="", use_annotation=True):
    is_star = False
    agg_id, col_id, isDistinct = col_unit1

    col_ful_name = schema.nameMap[col_id]
    table_name = col_ful_name.split(
        '.')[0] if col_ful_name != '*' else star_label.split('##')[0]
    col_name = col_ful_name.split('.')[1] if col_ful_name != '*' else '*'
    # To check if the column is date-related column
    is_year_col = False
    table_id = -1
    if table_name in table_dict['table_names_original']:
        table_id = table_dict['table_names_original'].index(table_name)
    index = -1
    for idx, col in enumerate(table_dict['column_names_original']):
        assert(len(col) == 2)
        t_id = col[0]
        cname = col[1].lower()
        if t_id == table_id and cname == col_name:
            index = idx
            break
    if index != -1:
        col_type = table_dict['column_types'][index]
        if col_type == 'time':
            is_year_col = True

    col_name_node = ""
    exist = False
    if edge_type == "gamma":
        # add a new edge between the existing nodes if the column is the same with the one in select clause;
        # otherwise, add a new column as normal.
        for col in sel_cols:
            sel_col_name = col.split('##')[0]
            if col_ful_name == sel_col_name:
                col_name_node = col
                exist = True
                break
    if not col_name_node:
        col_name_node = col_ful_name + '##' + \
            str(node_posttag_dict[col_ful_name])

    if col_ful_name == '*':
        is_star = True
        col_ful_name = star_label.split('##')[-1]
    # if use an existing node, we need to use the counting number before incrementing
    table_name_node = table_name + '##' + \
        str(node_posttag_dict[table_name] - 1)
    if not exist:
        # # Check if the belonging table is relationship type.
        # # If not, we assign two labels for primary attribute, one for projection, one for predicate.
        # key = f"{table_name}:type"
        # relationship = True if key in table_dict["annotations"] and table_dict["annotations"][key] == "relationship" else False
        if col_ful_name in table_dict['primaries']:
            G.add_node(
                col_name_node, type="attribute_node", primary=True,
                label=table_dict['annotations'][col_ful_name] if use_annotation and 'annotations' in table_dict.keys() and col_ful_name in table_dict['annotations'].keys() else s_strip(
                    col_ful_name),
                label1=table_dict['annotations'][table_name] if use_annotation and 'annotations' in table_dict.keys(
                ) and table_name in table_dict['annotations'].keys() else table_name
            )
        else:
            G.add_node(
                col_name_node, type="attribute_node", primary=True if col_ful_name in table_dict['primaries'] else False,
                label=table_dict['annotations'][col_ful_name] if
                use_annotation and 'annotations' in table_dict.keys(
                ) and col_ful_name in table_dict['annotations'].keys()
                else s_strip(col_ful_name)
            )
    key = table_name + '##' + col_ful_name
    if reverse:
        G.add_edge(
            col_name_node, table_name_node, type=edge_type,
            label=table_dict['annotations'][key] if use_annotation and 'annotations' in table_dict.keys(
            ) and key in table_dict['annotations'].keys() else TEMPLATE[edge_type],
            conj_name=conj
        )
    else:
        G.add_edge(
            table_name_node, col_name_node, type=edge_type,
            label=table_dict['annotations'][key] if use_annotation and 'annotations' in table_dict.keys(
            ) and key in table_dict['annotations'].keys() else TEMPLATE[edge_type],
            conj_name=conj
        )

    if isDistinct:
        distinct_node = "distinct" + '##' + str(node_posttag_dict["distinct"])
        G.add_node(
            distinct_node, type="distinct_node", primary=False,
            label=table_dict['annotations']["distinct"] if use_annotation and 'annotations' in table_dict.keys() and "distinct" in table_dict['annotations'].keys() else TEMPLATE[
                "distinct"]
        )
        G.add_edge(
            distinct_node, col_name_node, type="distnt",
            label=table_dict['annotations']["distinct"] if use_annotation and 'annotations' in table_dict.keys() and "distinct" in table_dict['annotations'].keys() else TEMPLATE[
                "distnt"]
        )
        node_posttag_dict["distinct"] += 1

    agg_opt = AGG_OPS[agg_id]
    if agg_opt != 'none':
        agg_opt_node = agg_opt + '##' + str(node_posttag_dict[agg_opt])
        G.add_node(
            agg_opt_node, type="function_node", primary=False,
            label=table_dict['annotations'][agg_opt] if use_annotation and 'annotations' in table_dict.keys() and agg_opt in table_dict['annotations'].keys() else TEMPLATE[
                agg_opt]
        )
        G.add_edge(
            agg_opt_node, col_name_node, type="r",
            label=table_dict['annotations'][agg_opt] if use_annotation and 'annotations' in table_dict.keys() and agg_opt in table_dict['annotations'].keys() else TEMPLATE["r"])
        node_posttag_dict[agg_opt] += 1

    if not is_star:
        node_posttag_dict[col_ful_name] += 1
    else:
        node_posttag_dict["*"] += 1

    return col_name_node, is_year_col


def build_graph_from_sql(G, sql_dict, table_dict, schema, parent_op=None, parent_node=None):
    # Read USE_ANNOTATION flag from configs.config to determine if using annotations for dialect generation.
    # use_annotation = USE_ANNOTATION
    use_annotation = True
    node_posttag_dict: Dict[str, int] = defaultdict(int)
    # We do not need query object currently, just random one of tables.
    query_obj = ""
    # We use star_label to represent the meaning of star in current query.
    # The format is "TABLE_NAME##LABEL_NAME".
    # As star actually does not link to a specific table, TABLE_NAME is only used to link to a node, the node is random node under joining.
    star_label = ""
    pk_fk_dict = defaultdict(set)
    # primary_keys = [pk for pk in table_dict["primary_keys"]]
    # pricol2table = {}
    # for k, col in enumerate(table_dict["column_names_original"][1:]):
    #     if (k+1) in primary_keys:
    #         pricol2table[k+1] = col[0]

    from_dict = sql_dict['from']
    tables = []
    # # Get the primary table index
    # primary_table = -1
    # for cond in tables['conds']:
    #     _, join_opt, val_unit1, val_unit2, _ = cond
    #     _, col_unit1, _ = val_unit1
    #     _, col_id1, _ = col_unit1
    #     _, col_id2, _ = val_unit2
    #     if all( k in pricol2table.keys()  for k in [col_id1, col_id2]):
    #         key = f"{table_dict['table_names_original'][pricol2table[col_id1]]}:type"
    #         if key in table_dict["annotations"] and table_dict["annotations"][key] == "relationship":
    #             primary_table = pricol2table[col_id2]
    #         else:
    #             primary_table = pricol2table[col_id1]
    #     elif col_id1 in pricol2table.keys(): primary_table = pricol2table[col_id1]
    #     elif col_id2 in pricol2table.keys(): primary_table = pricol2table[col_id2]
    #     else:
    #         print(f"The relation has no primary table!")

    # Add relation nodes and the corresponding edges
    if len(from_dict['table_units']) == 1:
        _, table_id = from_dict['table_units'][0]
        table_name = schema.nameMap[table_id]
        table_name_node = table_name + '##' + \
            str(node_posttag_dict[table_name])
        tables.append(table_name)
        G.add_node(
            table_name_node, type="relation_node", primary=False,
            label=table_dict['annotations'][table_name] if use_annotation and 'annotations' in table_dict.keys() and table_name in table_dict[
                'annotations'].keys() else table_name
        )
        node_posttag_dict[table_name] += 1
        query_obj = table_name_node
        star_label = f"{table_name}##{table_dict['annotations'][table_name]}" \
            if use_annotation and 'annotations' in table_dict.keys() and table_name in table_dict['annotations'].keys() else f"{table_name}##{table_name}"
    # If join exists
    else:
        # relation = '??'.join(sorted([schema.nameMap[t[1]] for t in from_dict['table_units']]))
        for i, (t1, t2) in enumerate(
                zip(from_dict['table_units'][:-1], from_dict['table_units'][1:])):
            _, table_id1 = t1
            table_name1 = schema.nameMap[table_id1]
            if i == 0:
                table_name_node1 = table_name1 + '##' + \
                    str(node_posttag_dict[table_name1])
                tables.append(table_name1)
            else:
                table_name_node1 = table_name1 + '##' + \
                    str(node_posttag_dict[table_name1] - 1)

            _, table_id2 = t2
            table_name2 = schema.nameMap[table_id2]
            if table_name1 == table_name2:
                if i == 0:
                    table_name_node2 = table_name2 + '##' + \
                        str(node_posttag_dict[table_name2] + 1)
                else:
                    table_name_node2 = table_name2 + '##' + \
                        str(node_posttag_dict[table_name2])
            else:
                table_name_node2 = table_name2 + '##' + \
                    str(node_posttag_dict[table_name2])
            tables.append(table_name2)

            if i == 0:
                label = table_name1
                # # If the table is relationship type, we assign the label for this table under current relation.
                # key = f"{table_name1}({relation})"
                # if key in table_dict["annotations"].keys():
                #     label = table_dict["annotations"][key]
                if use_annotation and 'annotations' in table_dict.keys() and table_name1 in table_dict['annotations'].keys():
                    label = table_dict['annotations'][table_name1]
                G.add_node(table_name_node1, type="relation_node",
                           primary=False, label=label)
                node_posttag_dict[table_name1] += 1

            label = table_name2
            # # If the table is relationship type, the label should be changed under different relations.
            # # It should exist an annotation which the format is as below,
            # # " [table_name](XXX??XXX) "
            # key = f"{table_name2}({relation})"
            # if key in table_dict["annotations"].keys():
            #     label = table_dict["annotations"][key]
            if use_annotation and 'annotations' in table_dict.keys() and table_name2 in table_dict['annotations'].keys():
                label = table_dict['annotations'][table_name2]
            G.add_node(table_name_node2, type="relation_node",
                       primary=False, label=label)
            node_posttag_dict[table_name2] += 1

            label = ""
            if from_dict['conds']:
                cond = from_dict['conds'][i*2]
                _, op_id, val_unit1, val_unit2, _ = cond
                _, col_unit1, _ = val_unit1
                _, col_id1, _ = col_unit1
                _, col_id2, _ = val_unit2

                col_name1 = schema.nameMap[col_id1]
                col_name2 = schema.nameMap[col_id2]
                pk_fk_dict[col_name1.split('.')[0]].add(col_name2.split('.')[0])
                pk_fk_dict[col_name2.split('.')[0]].add(col_name1.split('.')[0])
                # In order to exact match the annotation, we construct the sorted span into string
                labels = [col_name1, col_name2]
                labels.sort()
                label = WHERE_OPS[op_id].join(labels)
            # Hard-code for "FROM A JOIN B ON A.c1=B.c1 AND/OR A.c1 = B.c2" case
            if len(from_dict['table_units']) == len(from_dict['conds'][::2]) and len(from_dict['table_units']) == 2:
                _, op_id1, val_unit11, val_unit22, _ = from_dict['conds'][2]
                _, col_unit11, _ = val_unit11
                _, col_id11, _ = col_unit11
                _, col_id22, _ = val_unit22

                col_name11 = schema.nameMap[col_id11]
                col_name22 = schema.nameMap[col_id22]
                pk_fk_dict[col_name11.split('.')[0]].add(
                    col_name22.split('.')[0])
                pk_fk_dict[col_name22.split('.')[0]].add(
                    col_name11.split('.')[0])
                # In order to exact match the annotation, we construct the sorted span into string
                labels1 = [col_name11, col_name22]
                labels1.sort()
                label1 = WHERE_OPS[op_id1].join(labels1)
                label += '+' + label1
            G.add_edge(table_name_node1, table_name_node2,
                       type="theta", label=label)
            # label = f"{schema.nameMap[col_id2]}{WHERE_OPS[op_id]}{schema.nameMap[col_id1]}"
            G.add_edge(table_name_node2, table_name_node1,
                       type="theta", label=label)
    # Assign query object and star query object based on annotations
    # In order to align with the key string in annotations, we sort the tables alphabetically.
    if len(tables) > 1:
        tables.sort()
        # qo_key = '<>'.join(tables)
        # if qo_key not in table_dict["annotations"]:
        #     print(f"The query object annotation for this join {qo_key} not existing! Please check...")
        # qo_tbl = table_dict["annotations"][qo_key]
        query_obj = tables[0] + '##' + str(node_posttag_dict[tables[0]] - 1)
        star_qo_key = ''
        if use_annotation:
            star_qo_key = '*(' + '??'.join(tables) + ')'
            if 'annotations' in table_dict.keys() and star_qo_key not in table_dict["annotations"] and sql_dict['groupBy']:
                for col_unit in sql_dict['groupBy']:
                    _, col_id, _ = col_unit
                    col_name = schema.nameMap[col_id]
                    tab_name = col_name.split('.')[0]

                    if (star_qo_key + f"<{tab_name}>") not in table_dict["annotations"]:
                        for tname in pk_fk_dict[tab_name]:
                            if star_qo_key + f"<{tname}>" in table_dict["annotations"]:
                                tab_name = tname
                                break
                # cnt = 0
                # # If grouping on FK, the semantics should be the same with the same with PK.
                # while star_qo_key + f"<{tab_name}>" not in table_dict["annotations"] and ('id' in col_name or 'f1' in col_name) and cnt < len(tables):
                #     tab_name = tables[cnt]
                #     cnt += 1
                # if star_qo_key + f"<{tab_name}>" not in table_dict["annotations"] and cnt == len(tables): return None, query_obj
                star_qo_key += f"<{tab_name}>"
                # assert(star_qo_key in table_dict["annotations"])
        # if star_qo_key not in table_dict["annotations"]:
        #     print(f"The star query object annotation for this join {star_qo_key} not existing! Please check...")

        # if star_qo_key not in table_dict["annotations"]:
        #     print(star_qo_key)

        star_label = f"{tables[0]}##{table_dict['annotations'][star_qo_key]}" \
            if use_annotation and 'annotations' in table_dict.keys() and star_qo_key in table_dict["annotations"] else f"{tables[0]}##UNK"
        # star_query_obj = star_qo_tbl + '##' + str(node_posttag_dict[star_qo_tbl] - 1)

    # # if join exists
    # if len(tables) > 1:
    #     table1 = tables[0]
    #     table2 = tables[1]
    #     key1 = s_strip(table1) + '==' + s_strip(table2)
    #     G.add_edge(
    #             table1, table2, type="theta",
    #             label = table_dict['annotations'][key1] if key1 in table_dict['annotations'].keys() else TEMPLATE["join"]
    #         )
    #     key2 = s_strip(table2) + '==' + s_strip(table1)
    #     G.add_edge(
    #             table2, table1, type="theta",
    #             label = table_dict['annotations'][key2] if key2 in table_dict['annotations'].keys() else TEMPLATE["join"]
    #         )
    # _, _, val_unit, val1, _ = tables['conds']
    # _, col_unit1, _ = val_unit
    # col_name_node = build_graph_from_col_unit(G, col_unit1, table_dict, schema, node_posttag_dict, reverse=True)
    # col_name_node = build_graph_from_col_unit(G, val1, table_dict, schema, node_posttag_dict, reverse=True)

    # save the columns in select clause for groupby clause use
    sel_cols = []
    # TODO DISTINCT keyword in SELECT processing
    isDistinct = sql_dict['select'][0]
    projections = sql_dict['select'][1]
    for i, s in enumerate(projections):
        agg_id, val_unit = s
        # TODO col_unit2 processing
        _, col_unit1, _ = val_unit
        col_name_node, _ = build_graph_from_col_unit(G, col_unit1, table_dict, schema, node_posttag_dict, star_label,
                                                     reverse=True, use_annotation=use_annotation)
        sel_cols.append(col_name_node)
        # For the first project, check DISTINCT property out of col unit.
        if i == 0 and isDistinct:
            distinct_node = "distinct" + '##' + \
                str(node_posttag_dict["distinct"])
            G.add_node(
                distinct_node, type="distinct_node", primary=False,
                label=table_dict['annotations']["distinct"] if use_annotation and 'annotations' in table_dict.keys() and "distinct" in table_dict['annotations'].keys() else
                TEMPLATE["distinct"]
            )
            G.add_edge(
                distinct_node, col_name_node, type="distnt",
                label=table_dict['annotations']["distinct"] if use_annotation and 'annotations' in table_dict.keys() and "distinct" in table_dict['annotations'].keys() else
                TEMPLATE["distnt"]
            )
            node_posttag_dict["distinct"] += 1
        agg_opt = AGG_OPS[agg_id]
        if agg_opt != 'none':
            agg_opt_node = agg_opt + '##' + str(node_posttag_dict[agg_opt])
            G.add_node(
                agg_opt_node, type="function_node", primary=False,
                label=table_dict['annotations'][agg_opt] if use_annotation and 'annotations' in table_dict.keys() and agg_opt in table_dict['annotations'].keys() else TEMPLATE[
                    agg_opt]
            )
            G.add_edge(
                agg_opt_node, col_name_node, type="r",
                label=table_dict['annotations'][agg_opt] if use_annotation and 'annotations' in table_dict.keys() and agg_opt in table_dict['annotations'].keys() else TEMPLATE[
                    "r"]
            )
            node_posttag_dict[agg_opt] += 1

        # # if it is in subquery, connect the selection node with the predicate node which belongs to the main relation
        # if parent_node:
        #     G.add_edge(
        #         parent_node, col_name_node, type=parent_op,
        #         label = table_dict['annotations'][parent_op] if parent_op in table_dict['annotations'].keys() else TEMPLATE[parent_op]
        #     )

    if sql_dict['where']:
        predicates = sql_dict['where']
        # For first predicate, there is no conjunction string along with it.
        conjs = [""] + [s for s in predicates[1::2]]
        for cond_unit, conj in zip(predicates[::2], conjs):
            G = build_graph_from_cond_unit(G, cond_unit, conj, table_dict, schema, node_posttag_dict, star_label,
                                           edge_type="theta", use_annotation=use_annotation)

    if sql_dict['groupBy']:
        for col_unit in sql_dict['groupBy']:
            build_graph_from_col_unit(G, col_unit, table_dict, schema, node_posttag_dict, star_label, edge_type="gamma",
                                      sel_cols=sel_cols, use_annotation=use_annotation)
        if sql_dict['having']:
            conditions = sql_dict['having']
            if sql_dict['where']:
                conjs = ["and"] + [s for s in conditions[1::2]]
            else:
                conjs = [""] + [s for s in conditions[1::2]]
            for cond_unit, conj in zip(conditions[::2], conjs):
                G = build_graph_from_cond_unit(G, cond_unit, conj, table_dict, schema, node_posttag_dict, star_label,
                                               edge_type="h", use_annotation=use_annotation)

    if sql_dict['orderBy']:
        ordering_term, val_units = sql_dict['orderBy']
        if ordering_term == 'asc':
            edge_type = 'ao'
        elif ordering_term == 'desc':
            edge_type = 'do'
        else:
            edge_type = 'o'
        for val_unit in val_units:
            _, col_unit1, _ = val_unit
            col_name_node, _ = build_graph_from_col_unit(G, col_unit1, table_dict, schema, node_posttag_dict, star_label,
                                                         edge_type=edge_type, use_annotation=use_annotation)

    # processing LIMIT
    if sql_dict['limit']:
        limit_n = sql_dict['limit']
        # Hard-code to fix masked-out-LIMIT in the extracted nested-predicate unit
        if limit_n == '"value"':
            limit_n = 1
        limit_n_node = str(limit_n) + '##' + \
            str(node_posttag_dict[str(limit_n)])
        G.add_node(limit_n_node, type="value_node", primary=False, value=limit_n,
                   label=TEMPLATE[str(limit_n).strip('"')])
        node_posttag_dict[str(limit_n)] += 1
        G.add_edge(
            query_obj, limit_n_node, type='limit',
            label=table_dict['annotations']['limit'] if use_annotation and 'annotations' in table_dict.keys() and 'limit' in table_dict['annotations'].keys() else TEMPLATE[
                'limit']
        )

    return G, query_obj
