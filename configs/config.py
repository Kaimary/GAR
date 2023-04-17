import os
DIR_PATH=os.getcwd() 
# Query Unit Knobs
QUNITS_FILE = '/serialization/gar.{0}/qunits.json'
QUNITS_DEBUG_FILE = '/serialization/gar.{0}/qunits_debug.json'
QUNITS_SET_COVER_FILE = '/serialization/gar.{0}/qunits_set_cover.json'
QUNITS_SET_COVER_MINUS_FILE = '/serialization/gar.{0}/qunits_set_generated_cover.json'
# Retrieval Model Knobs
RETRIEVAL_MODEL_TRAIN_BATCH_SIZE=15
RETRIEVAL_MODEL_DIMENSION = 768
RETRIEVAL_MODEL_NAME = 'stsb-mpnet-base-v2'
RETRIEVAL_MODEL_DIR = '/saved_models/gar.{0}/retrieval_model'
RETRIEVAL_MODEL_TRAIN_DATA_GZ_FILE = '/saved_models/gar.{0}/retrieval_model/train_dev.tsv.gz'
RETRIEVAL_MODEL_DATA_FINETUNE_GZ_FILE = '/saved_models/gar.{0}/retrieval_model/finetune.tsv.gz'
# Default lr is 2e-5 
RETRIVIAL_MODEL_LEARNING_RATE = 5e-6
CANDIDATE_NUM=100
# Re-ranker Model Knobs
RERANKER_EMBEDDING_MODEL_NAME='roberta-base'
RERANKER_MODEL_NAME='bertpooler'
RERANKER_TRAIN_DATA_FILE = '/saved_models/gar.{0}/reranker/train.json'
RERANKER_DEV_DATA_FILE = '/saved_models/gar.{0}/reranker/dev.json'
RERANKER_CONFIG_FILE = '/ranker/BERTPooler/train_configs/ranker_pair.jsonnet'
RERANKER_MODEL_DIR= '/saved_models/gar.{0}/reranker'
# Serialization knobs
GENERATION_NUM=20000
SERIALIZE_DATA_DIR = '/serialization/gar.{0}'
# Output knobs
OUTPUT_DIR='/output/data={0},num1={1},num2={2},model1={3},model2={4}'
PRED_FILE_NAME='pred.txt'
PRED_SQL_FILE_NAME='pred_sql.txt'
PRED_TOPK_FILE_NAME='pred_topk.txt'
PRED_SQL_TOPK_FILE_NAME='pred_sql_topk.txt'
RERANKER_MISS_FILE_NAME='reranker_miss.txt'
RERANKER_MISS_TOPK_FILE_NAME='reranker_miss_topk.txt'
VALUE_FILTERED_TOPK_FILE_NAME='value_filtered_miss_topk.txt'
# Misc.
# SQL gen time limit
TIME_LIMIT_PRE_SQL = 5  # seconds generate more than one sql
MODEL_TAR_GZ='model.tar.gz'
RERANKER_DEV_DATA_MAX_NUM = 1000
TOP_NUM=10
# re-generate serialized dialects 
# (True to make any change in Dialect Builder gets effected)
REWRITE_FLAG=False
# Overwrite flag for the serialized generation data 
# (True to make any change in SQLGenV2 gets effected)
OVERWRITE_FLAG=False
SQLGEN_DEBUG_FLAG=True
RETREVAL_DEBUG_FLAG=True
# Toggle to enable/disable using annotations during dialect generation
USE_ANNOTATION=False