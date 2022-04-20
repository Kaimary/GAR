from value_mathcing.spider_db_context import SpiderDBContext, is_number
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from spider_utils.evaluation.evaluate import rebuild_sql, build_foreign_key_map_from_json
from datagen.sqlgen.utils.sql_tmp_update import sql_nested_query_tmp_name_convert
import json
import copy
from tqdm import tqdm
import click


def all_number(cur_value_set):
    flag = True
    for i in cur_value_set:
        if not is_number(i):
            flag = False
    return flag


def nl_reader(nl_file_path):
    nl_list = []
    db_id_list = []
    with open(nl_file_path, 'r') as f_in:
        json_obj = json.load(f_in)
        for ex in json_obj:
            nl_list.append(ex['question'])
            db_id_list.append(ex['db_id'])
    return nl_list, db_id_list


def candidate_reader(candidates_file_path):
    candidate_list = []
    with open(candidates_file_path, 'r') as f_in:
        tmp = []
        for line in f_in.readlines():
            line = line.strip()
            if line == '':
                candidate_list.append(copy.deepcopy(tmp))
                tmp.clear()
            else:
                tmp.append(line)
        if len(tmp) != 0:
            candidate_list.append(copy.deepcopy(tmp))
    return candidate_list


def get_all_filter_column(sql_dict):
    """
    get all filter conditions from one sql dict
    @param sql_dict: sql rebuilt
    @return: filter columns set
    """
    # no a sql dict, no column found
    if not isinstance(sql_dict, dict):
        return {}
    filter_columns = list()
    # add every condition's col into set
    for condition in sql_dict['where']:
        # pass conjunction
        if condition == 'and' or condition == 'or':
            continue
        else:
            filter_columns.append(condition[2][1][1].strip('_'))  # add column
            filter_columns.extend(get_all_filter_column(condition[3]))  # recursively add column from nested
    # recursively handle IUE
    filter_columns.extend(get_all_filter_column(sql_dict['intersect']))
    filter_columns.extend(get_all_filter_column(sql_dict['union']))
    filter_columns.extend(get_all_filter_column(sql_dict['except']))
    return filter_columns


def candidate_filter(candidates, db_id, db_context, tables_file, dataset_path):
    cur_column_list = []
    cur_value_set = set()

    entities = db_context.get_entities_from_question(db_context.string_column_mapping)
    cur_value_set.clear()
    cur_column_list.clear()

    for ent in entities:
        if is_number(ent[0].split(':')[-1]) or ent[0].split(':')[-1] == '' or ent[0].split(':')[-1] == 'd':
            continue
        cur_value_set.add(ent[0].split(':')[-1])
        col_tmp = set()
        for col in ent[1]:
            col_tmp.add(col.split(':')[-2] + '.' + col.split(':')[-1])
        cur_column_list.append(col_tmp)

    if cur_column_list:
        cur_candidates = []
        cur_del_num = 0
        del_index_list = []
        kmaps = build_foreign_key_map_from_json(tables_file)

        for candidate in candidates:
            append_flag = True

            sql_dict = rebuild_sql(db_id, dataset_path, sql_nested_query_tmp_name_convert(candidate), kmaps)
            sql_filter_cols = get_all_filter_column(sql_dict)
            for col_set in cur_column_list:
                check_col_flag = False
                for col in col_set:
                    if col in sql_filter_cols:
                        check_col_flag = True
                        sql_filter_cols.remove(col)
                        break
                if not check_col_flag:
                    append_flag = False
                    break
            if append_flag:
                cur_candidates.append(candidate)
            else:
                del_index_list.append(candidates.index(candidate))
                cur_del_num += 1
        if cur_del_num == 10:
            cur_candidates = candidates
        return cur_candidates
    else:
        return candidates


@click.command()
@click.argument("nl_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidates_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("database_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("pred_sql_path", type=click.Path(exists=False, dir_okay=False))
@click.argument("pred_sql_path_top_k", type=click.Path(exists=False, dir_okay=False))
def main(nl_file_path, candidates_file_path, tables_file, database_dir, pred_sql_path, pred_sql_path_top_k):
    tokenizer = SpacyTokenizer()
    # definition
    pred_sql_list = []

    # data reader
    # 1.nl, db_id
    nl_list, db_id_list = nl_reader(nl_file_path)

    # 2.candidate list
    candidate_list = candidate_reader(candidates_file_path)

    # filter loop, return pred sql list
    db_context = SpiderDBContext(
        db_id_list[0],
        nl_list[0],
        tokenizer,
        tables_file,
        database_dir
    )
    for nl, candidates, db_id in tqdm(zip(nl_list, candidate_list, db_id_list)):
        if db_context.db_id != db_id:
            db_context = SpiderDBContext(
                db_id,
                nl,
                tokenizer,
                tables_file,
                database_dir
            )
        db_context.change_utterance(nl)
        if nl == "What is the total surface area of the continents Asia and Europe?":
            print("Here")
        tmp = candidate_filter(candidates, db_id, db_context, tables_file, database_dir)
        # padding to 10 sqls
        for i in range(10 - len(tmp)):
            tmp.append(tmp[-1])
        pred_sql_list.append(tmp)

    # 3.pred sql writer
    with open(pred_sql_path, 'w') as f_out:
        for sqls in pred_sql_list:
            f_out.write(sqls[0] + '\n')
    with open(pred_sql_path_top_k, 'w') as f_out_1:
        for sqls in pred_sql_list:
            for sql in sqls:
                f_out_1.write(sql + '\n')
            f_out_1.write('\n')


def test():
    nl_file_path = "sqlgenv2/datasets/spider/dev.json"
    candidates_file_path = "sqlgenv2/output/spider/reranker/spider_20000_100_stsb-mpnet-base-v2_roberta-base_bertpooler/pred_sql_topk.txt"
    tables_file = "sqlgenv2/datasets/spider/tables.json"
    database_dir = "sqlgenv2/datasets/spider/database"
    pred_sql_path = "sqlgenv2/output/spider/reranker/spider_20000_100_stsb-mpnet-base-v2_roberta-base_bertpooler/pred_sql_value_filtered.txt"
    pred_sql_path_top_k = "sqlgenv2/output/spider/reranker/spider_20000_100_stsb-mpnet-base-v2_roberta-base_bertpooler/pred_sql_topk_value_filtered.txt"
    main(nl_file_path, candidates_file_path, tables_file, database_dir, pred_sql_path, pred_sql_path_top_k)


if __name__ == '__main__':
    # test()
    main()
