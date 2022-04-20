import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

config_vars = {}
with open(dir_path+'/config.py', 'r') as f:
    for line in f:
        if '=' in line:
            k,v = line.split('=', 1)
            k = k.strip()
            if k in ["SENTENCE_EMBEDDER_MODEL_DIMENSION", "SENTENCE_EMBEDDER_BASE_MODEL_NAME", "TEST_DATA_FILE_NAME", "PRED_FILE_NAME", "PRED_TOPK_FILE_NAME", "CANDIDATE_MISS_FILE_NAME", "SQL_MISS_FILE_NAME", "RERANKER_MISS_FILE_NAME", "MODEL_TAR_GZ"]:
                config_vars[k] = v.strip().strip("'")
            elif k in ['OUTPUT_DIR_RERANKER', 'RERANKER_MODEL_DIR']:
                config_vars[k] = dir_path + v.format(sys.argv[1]).strip().strip("'")
            else:
                config_vars[k] = dir_path + v.strip().strip("'")
#print(f"config_vars:{config_vars}")
print(f"{config_vars['OUTPUT_DIR_RERANKER']}@{config_vars['RERANKER_MODEL_DIR']}@{config_vars['TEST_DATA_FILE_NAME']}@{config_vars['PRED_FILE_NAME']}@{config_vars['RERANKER_MISS_FILE_NAME']}@{config_vars['MODEL_TAR_GZ']}@{config_vars['PRED_TOPK_FILE_NAME']}")
