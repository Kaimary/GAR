import os
import sys
dir_path = os.getcwd() 

config_vars = {}
with open(dir_path+'/configs/config.py', 'r') as f:
    for line in f:
        if '=' in line:
            k,v = line.split('=', 1)
            k = k.strip()
            if k in ["RETRIEVAL_MODEL_NAME", "RERANKER_EMBEDDING_MODEL_NAME", "RERANKER_MODEL_NAME", \
                "RETRIEVAL_MODEL_DIMENSION", "PRED_FILE_NAME", "PRED_TOPK_FILE_NAME", "MODEL_TAR_GZ", \
                    "GENERATION_NUM", "CANDIDATE_NUM"]:
                config_vars[k] = v.strip().strip("'")
            elif k == 'OUTPUT_DIR':
                config_vars[k] = dir_path + \
                    v.format(
                        sys.argv[1], 
                        config_vars['GENERATION_NUM'], 
                        config_vars['CANDIDATE_NUM'], 
                        config_vars['RETRIEVAL_MODEL_NAME'], 
                        f"{config_vars['RERANKER_MODEL_NAME']}+{config_vars['RERANKER_EMBEDDING_MODEL_NAME']}"
                    ).strip().strip("'")
            elif k in ['RERANKER_MODEL_DIR']:
                config_vars[k] = dir_path + v.format(sys.argv[1]).strip().strip("'")
            else:
                config_vars[k] = dir_path + v.strip().strip("'")
print(f"{config_vars['OUTPUT_DIR']}@{config_vars['RERANKER_MODEL_DIR']}@{config_vars['PRED_FILE_NAME']}@{config_vars['MODEL_TAR_GZ']}@{config_vars['PRED_TOPK_FILE_NAME']}@{config_vars['RERANKER_MODEL_NAME']}@{config_vars['RERANKER_EMBEDDING_MODEL_NAME']}")
