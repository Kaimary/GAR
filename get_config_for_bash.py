import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

config_vars = {}
with open(dir_path + '/config.py', 'r') as f:
    for line in f:
        if '=' in line:
            k, v = line.split('=', 1)
            k = k.strip()
            if k in ["SENTENCE_EMBEDDER_MODEL_DIMENSION", "SENTENCE_EMBEDDER_BASE_MODEL_NAME"]:
                config_vars[k] = v.strip().strip("'")
            elif k in ['SEMSIMILARITY_TRIPLE_DATA_GZ_FILE', 'RERANKER_TRAIN_DATA_FILE', 'RERANKER_DEV_DATA_FILE', 'RERANKER_MODEL_DIR', 'SENTENCE_EMBEDDER_MODEL_DIR']:
                config_vars[k] = dir_path + v.format(sys.argv[1]).strip().strip("'")
            else:
                config_vars[k] = dir_path + v.strip().strip("'")
# print(f"config_vars:{config_vars}")
print(
    f"{config_vars['SEMSIMILARITY_TRIPLE_DATA_GZ_FILE']}@{config_vars['SENTENCE_EMBEDDER_MODEL_DIR']}@"
    f"{config_vars['RERANKER_TRAIN_DATA_FILE']}@{config_vars['RERANKER_DEV_DATA_FILE']}@{config_vars['RERANKER_CONFIG_FILE']}@"
    f"{config_vars['RERANKER_MODEL_DIR']}"
)
