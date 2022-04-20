#!/bin/bash

[ -z "$1" ] && echo "First argument should be the name of the dataset." && exit 1
DATASET_NAME="$1"

[ -z "$2" ] && echo "Second argument should be the dataset dev file." && exit 1
DATASET_DEV_FILE="$2"

[ -z "$3" ] && echo "Third argument should be the schema file of the dataset." && exit 1
TABLES_FILE="$3"

[ -z "$4" ] && echo "Fourth argument should be the directory of the databases for the dataset." && exit 1
DB_DIR="$4"

[ -z "$5" ] && echo "Fifth argument should be the size of SQL set that needs to be generated." && exit 1
NUM_SQL="$5"

[ -z "$6" ] && echo "Sixth argument should be the number of candidate SQLs that needs to be selected from SQL set first." && exit 1
CANDIDATE_NUM="$6"

[ -z "$7" ] && echo "Seventh argument should be the overwrite flag. " && exit 1
OVERWRITE="$7"

[ -z "$8" ] && echo "Eighth argument should be candidate base word embedding model name. " && exit 1
CANDIDATE_EMBEDDING_MODEL_NAME="$8"

[ -z "$9" ] && echo "Ninth argument should be reranking base word embedding model name. " && exit 1
RERANKER_EMBEDDING_MODEL_NAME="$9"

[ -z "${10}" ] && echo "Tenth argument should be reranking model name. " && exit 1
RERANKER_MODEL_NAME="${10}"

[ -z "${11}" ] && echo "Eleventh argument should be dev gold sql file used for evaluation. " && exit 1
DEV_GOLD="${11}"

[ -z "${12}" ] && echo "Twelfth argument should be if 82 setup is chosen." && exit 1
IS_82="${12}"


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DIR=$(pwd)

while true; do
    echo "Dataset name: $DATASET_NAME"
    echo "Dataset file(Validation): $DATASET_DEV_FILE"
    echo "Schema file of the dataset: $TABLES_FILE"
    echo "Databases directory of the dataset: $DB_DIR"
    echo "Number of SQLs that needs to be generated: $NUM_SQL"
    echo "Number of candidate SQLs that need to be selected first: $CANDIDATE_NUM"
    echo "Overwrite flag for the generated data(True only if the SQL generator has changed): $OVERWRITE"
    echo "Candidate base word embedding model name: $CANDIDATE_EMBEDDING_MODEL_NAME"
    echo "Reranking base word embedding model name: $RERANKER_EMBEDDING_MODEL_NAME"
    echo "Reranking model name: $RERANKER_MODEL_NAME"
    echo "Dev gold sql: $DEV_GOLD"
    echo "Is 82 setup: $IS_82"
    read -p "Is this ok [y/n] ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit 0;;
        * ) echo "Please answer y or n.";;
    esac
done

echo "=================================================================="
echo "ACTION REPORT: Testing pipeline starts ......"
output=`python3 -m get_config_for_test_bash $DATASET_NAME`
OUTPUT_DIR_RERANKER=$(cut -d'@' -f1 <<< "$output")
RERANKER_MODEL_DIR=$(cut -d'@' -f2 <<< "$output")
TEST_DATA_FILE_NAME=$(cut -d'@' -f3 <<< "$output")
PRED_FILE_NAME=$(cut -d'@' -f4 <<< "$output")
RERANKER_MISS_FILE_NAME=$(cut -d'@' -f5 <<< "$output")
MODEL_TAR_GZ=$(cut -d'@' -f6 <<< "$output")
PRED_TOPK_FILE_NAME=$(cut -d'@' -f7 <<< "$output")

EXPERIMENT_DIR_NAME=$OUTPUT_DIR_RERANKER/$DATASET_NAME\_$NUM_SQL\_$CANDIDATE_NUM\_$CANDIDATE_EMBEDDING_MODEL_NAME\_$RERANKER_EMBEDDING_MODEL_NAME\_$RERANKER_MODEL_NAME
RERANKER_TEST_DATA_FILE=$EXPERIMENT_DIR_NAME/$TEST_DATA_FILE_NAME
RERANKER_MODEL_FILE=$RERANKER_MODEL_DIR\_$RERANKER_MODEL_NAME\_$RERANKER_EMBEDDING_MODEL_NAME/$MODEL_TAR_GZ
RERANKER_MODEL_OUTPUT_FILE=$EXPERIMENT_DIR_NAME/$PRED_FILE_NAME
RERANKER_MODEL_OUTPUT_TOPK_FILE=$EXPERIMENT_DIR_NAME/$PRED_TOPK_FILE_NAME
RERANKER_MODEL_OUTPUT_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql.txt}
RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_topk.txt}
EVALUATE_OUTPUT_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_evaluate.txt}
VALUE_FILTERED_OUTPUT_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_value_filtered.txt}
VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE=${RERANKER_MODEL_OUTPUT_FILE/.txt/_sql_topk_value_filtered.txt}


# Generate test data
if [ ! -f $RERANKER_TEST_DATA_FILE ]; then
    echo "ACTION REPORT: Generate re-ranker test data into $RERANKER_TEST_DATA_FILE"
    python3 -m datagen.model_datagen.re-ranker_listwise_data_gen_for_test $DATASET_NAME $DATASET_DEV_FILE $TABLES_FILE $DB_DIR $NUM_SQL $CANDIDATE_NUM $OVERWRITE $EXPERIMENT_DIR_NAME $CANDIDATE_EMBEDDING_MODEL_NAME $IS_82
    echo "RESULT REPORT: Re-ranker test data is ready now!"
    echo "=================================================================="
else
    echo "Re-ranker test data exists!"
    echo "=================================================================="
fi


# Inference for top-1
if [ -f $RERANKER_MODEL_FILE -a ! -f $RERANKER_MODEL_OUTPUT_FILE ]; then
    echo "ACTION REPORT: Start to test re-ranker model $RERANKER_MODEL_FILE"
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_TEST_DATA_FILE" --file-friendly-logging --silent --predictor listwise-ranker --use-dataset-reader --cuda-device 0 --output-file "$RERANKER_MODEL_OUTPUT_FILE" --include-package dataset_readers.listwise_pair_reader --include-package models.semantic_matcher.listwise_pair_ranker --include-package predictors.ranker_predictor || exit $?
    echo "RESULT REPORT: Re-ranker model test complete!"
    echo "=================================================================="
else
    echo "Re-ranker model $RERANKER_MODEL_FILE does not exist or $RERANKER_MODEL_OUTPUT_FILE exists."
    echo "=================================================================="
fi


# Evaluate re-ranker model
if [ -f $RERANKER_MODEL_OUTPUT_FILE -a ! -f $RERANKER_MODEL_OUTPUT_SQL_FILE ]; then
    echo "ACTION REPORT: Start to evaluate re-ranker model ......"
    python3 -m spider_custom_evaluate_for_test $TABLES_FILE $DB_DIR $RERANKER_MODEL_OUTPUT_FILE $RERANKER_TEST_DATA_FILE $EXPERIMENT_DIR_NAME
    echo "RESULT REPORT: Re-ranker model evaluation complete!"
    echo "=================================================================="
else
    echo "Re-ranker output does not exist or top1 sql file exists"
    echo "=================================================================="
fi


# Inference for top-k
if [ -f $RERANKER_MODEL_FILE -a ! -f $RERANKER_MODEL_OUTPUT_TOPK_FILE ]; then
    echo "ACTION REPORT: Start to test re-ranker model $RERANKER_MODEL_FILE"
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_TEST_DATA_FILE" --file-friendly-logging --silent --predictor listwise-ranker --use-dataset-reader --cuda-device 0 --output-file "$RERANKER_MODEL_OUTPUT_TOPK_FILE" --include-package dataset_readers.listwise_pair_reader --include-package models.semantic_matcher.listwise_pair_ranker --include-package predictors.ranker_predictor_topk || exit $?
    echo "RESULT REPORT: Re-ranker model test (top-k) complete!"
    echo "=================================================================="
else
    echo "Re-ranker model $RERANKER_MODEL_FILE does not exist or $RERANKER_MODEL_OUTPUT_TOPK_FILE exists."
    echo "=================================================================="
fi


# Evaluate for top-k
if [ -f $RERANKER_MODEL_OUTPUT_TOPK_FILE -a ! -f $RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE ]; then
    echo "ACTION REPORT: Start to generate top-k result ......"
    python3 -m spider_custom_evaluate_topk $TABLES_FILE $DB_DIR $RERANKER_MODEL_OUTPUT_TOPK_FILE $RERANKER_TEST_DATA_FILE $EXPERIMENT_DIR_NAME
    echo "RESULT REPORT: Top-k result generate complete!"
    echo "=================================================================="
else
    echo "Re-ranker topk output does not exist or topk sql file exists"
    echo "=================================================================="
fi


# Value filtered
if [ -f $RERANKER_MODEL_OUTPUT_SQL_FILE -a ! -f $VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE ]; then
    echo "Value filter stage starting..."
    python3 -m value_mathcing.candidate_filter_top10 "$DATASET_DEV_FILE" "$RERANKER_MODEL_OUTPUT_TOPK_SQL_FILE" "$TABLES_FILE" "$DB_DIR" "$VALUE_FILTERED_OUTPUT_SQL_FILE" "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE"
    echo "Value filter result saved in $VALUE_FILTERED_OUTPUT_SQL_FILE"
    echo "Value filter top-k result saved in $VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE"
    echo "Value filter stage complete!"
    echo "=================================================================="
else
    echo "Value filter result exist!"
    echo "=================================================================="
fi


# Evaluate for value filtered
if [ -f $VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE ]; then
    echo "Value filter evaluation starting..."
    python3 -m value_mathcing.value_matching_evaluate "$TABLES_FILE" "$DB_DIR" "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" "$EXPERIMENT_DIR_NAME"
    echo "Value filter evaluation complete!"
    echo "=================================================================="
else
    echo "Value filter result exist!"
    echo "=================================================================="
fi


# Final Evaluation
if [ -f $VALUE_FILTERED_OUTPUT_SQL_FILE -a ! -f $EVALUATE_OUTPUT_FILE ]; then
    echo "Start evaluate"
    python3 -m spider_utils.evaluation.evaluate --gold "$DEV_GOLD" --pred "$VALUE_FILTERED_OUTPUT_SQL_FILE" --etype "match" --db "$DB_DIR" --table "$TABLES_FILE" --candidates "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" > "$EVALUATE_OUTPUT_FILE"
    echo "Evaluation Finished!"
    echo "Evaluation result saved in $EVALUATE_OUTPUT_FILE"
    echo "=================================================================="
else
    echo "Evaluation result exist!"
    echo "=================================================================="
fi

echo "Testing Pipeline completed!"
