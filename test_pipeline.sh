#!/bin/bash

[ -z "$1" ] && echo "First argument should be the name of the dataset." && exit 1
DATASET_NAME="$1"

[ -z "$2" ] && echo "Second argument is the test json file." && exit 1
TEST_FILE="$2"

[ -z "$3" ] && echo "Third argument is dev gold sql file used for evaluation. " && exit 1
GOLD_SQL_FILE="$3"

[ -z "$4" ] && echo "Fourth argument should be the schema file of the dataset." && exit 1
TABLES_FILE="$4"

[ -z "$5" ] && echo "Fifth argument should be the directory of the databases for the dataset." && exit 1
DB_DIR="$5"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DIR=$(pwd)

while true; do
    echo "Dataset name: $DATASET_NAME"
    echo "Test JSON file: $TEST_FILE"
    echo "Gold SQL TXT file: $GOLD_SQL_FILE"
    echo "Schema file of the dataset: $TABLES_FILE"
    echo "Databases directory of the dataset: $DB_DIR"
    read -p "Is this ok [y/n] ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit 0;;
        * ) echo "Please answer y or n.";;
    esac
done

echo "=================================================================="
echo "ACTION REPORT: Testing pipeline starts ......"
output=`python3 -m configs.get_config_for_test_bash $DATASET_NAME`
OUTPUT_DIR=$(cut -d'@' -f1 <<< "$output")
if [ ! -d $OUTPUT_DIR ]; then
  mkdir -p $OUTPUT_DIR;
fi
RERANKER_MODEL_DIR=$(cut -d'@' -f2 <<< "$output")
PRED_FILE_NAME=$(cut -d'@' -f3 <<< "$output")
MODEL_TAR_GZ=$(cut -d'@' -f4 <<< "$output")
PRED_TOPK_FILE_NAME=$(cut -d'@' -f5 <<< "$output")
RERANKER_MODEL_NAME=$(cut -d'@' -f6 <<< "$output")
RERANKER_EMBEDDING_MODEL_NAME=$(cut -d'@' -f7 <<< "$output")

RERANKER_INPUT_FILE="${OUTPUT_DIR}/test.json"
RERANKER_OUTPUT_FILE=$OUTPUT_DIR/$PRED_FILE_NAME
RERANKER_MODEL_FILE=$RERANKER_MODEL_DIR/$RERANKER_MODEL_NAME\+$RERANKER_EMBEDDING_MODEL_NAME/$MODEL_TAR_GZ
RERANKER_OUTPUT_TOPK_FILE=$OUTPUT_DIR/$PRED_TOPK_FILE_NAME
RERANKER_OUTPUT_SQL_FILE=${RERANKER_OUTPUT_FILE/.txt/_sql.txt}
RERANKER_OUTPUT_TOPK_SQL_FILE=${RERANKER_OUTPUT_FILE/.txt/_sql_topk.txt}
EVALUATE_OUTPUT_FILE=${RERANKER_OUTPUT_FILE/.txt/_evaluate.txt}
VALUE_FILTERED_OUTPUT_SQL_FILE=${RERANKER_OUTPUT_FILE/.txt/_sql_value_filtered.txt}
VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE=${RERANKER_OUTPUT_FILE/.txt/_sql_topk_value_filtered.txt}

# Generate test data for reranker
if [ ! -f $RERANKER_INPUT_FILE ]; then
    echo "ACTION REPORT: Generate re-ranker test data into $RERANKER_INPUT_FILE"
    python3 -m scripts.reranker_train_test_script $DATASET_NAME $TEST_FILE $TABLES_FILE $DB_DIR $OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Re-ranker test data generation failed!"
        exit;
    fi
    echo "RESULT REPORT: Re-ranker test data is ready now!"
    echo "=================================================================="
else
    echo "Re-ranker test data exists!"
    echo "=================================================================="
fi
# Inference for top-1
if [ -f $RERANKER_MODEL_FILE -a ! -f $RERANKER_OUTPUT_FILE ]; then
    echo "ACTION REPORT: Start to test re-ranker model $RERANKER_MODEL_FILE"
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_INPUT_FILE" --file-friendly-logging --silent --predictor \
    listwise-ranker --use-dataset-reader --cuda-device 0 --output-file "$RERANKER_OUTPUT_FILE" \
    --include-package ranker.BERTPooler.dataset_readers.listwise_pair_reader \
    --include-package ranker.BERTPooler.models.listwise_pair_ranker \
    --include-package ranker.BERTPooler.predictors.ranker_predictor || exit $?
    echo "RESULT REPORT: Re-ranker model test complete!"
    echo "=================================================================="
else
    echo "Re-ranker model $RERANKER_MODEL_FILE does not exist or $RERANKER_OUTPUT_FILE exists."
    echo "=================================================================="
fi


# Evaluate re-ranker model
if [ -f $RERANKER_OUTPUT_FILE -a ! -f $RERANKER_OUTPUT_SQL_FILE ]; then
    echo "ACTION REPORT: Start to evaluate re-ranker model ......"
    python3 -m scripts.evaluation.reranker_evaluate $TABLES_FILE $DB_DIR $RERANKER_OUTPUT_FILE $RERANKER_INPUT_FILE $OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Re-ranker model evaluation failed!"
        exit;
    fi
    echo "RESULT REPORT: Re-ranker model evaluation complete!"
    echo "=================================================================="
else
    echo "Re-ranker output does not exist or top1 sql file exists"
    echo "=================================================================="
fi


# Inference for top-k
if [ -f $RERANKER_MODEL_FILE -a ! -f $RERANKER_OUTPUT_TOPK_FILE ]; then
    echo "ACTION REPORT: Start to test re-ranker model $RERANKER_MODEL_FILE"
    allennlp predict "$RERANKER_MODEL_FILE" "$RERANKER_INPUT_FILE" --file-friendly-logging --silent \
    --predictor listwise-ranker --use-dataset-reader --cuda-device 0 --output-file "$RERANKER_OUTPUT_TOPK_FILE" \
    --include-package ranker.BERTPooler.dataset_readers.listwise_pair_reader \
    --include-package ranker.BERTPooler.models.listwise_pair_ranker \
    --include-package ranker.BERTPooler.predictors.ranker_predictor_topk || exit $?
    echo "RESULT REPORT: Re-ranker model test (top-k) complete!"
    echo "=================================================================="
else
    echo "Re-ranker model $RERANKER_MODEL_FILE does not exist or $RERANKER_OUTPUT_TOPK_FILE exists."
    echo "=================================================================="
fi


# Evaluate for top-k
if [ -f $RERANKER_OUTPUT_TOPK_FILE -a ! -f $RERANKER_OUTPUT_TOPK_SQL_FILE ]; then
    echo "ACTION REPORT: Start to generate top-k result ......"
    python3 -m scripts.evaluation.reranker_evaluate_topk $TABLES_FILE $DB_DIR $RERANKER_OUTPUT_TOPK_FILE $RERANKER_INPUT_FILE $OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Top-k result generate failed!"
        exit;
    fi
    echo "RESULT REPORT: Top-k result generate complete!"
    echo "=================================================================="
else
    echo "Re-ranker topk output does not exist or topk sql file exists"
    echo "=================================================================="
fi


# Value filtered
if [ -f $RERANKER_OUTPUT_SQL_FILE -a ! -f $VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE ]; then
    echo "Value filter stage starting..."
    python3 -m scripts.value_postprocessing.candidate_filter_top10 "$TEST_FILE" "$RERANKER_OUTPUT_TOPK_SQL_FILE" "$TABLES_FILE" "$DB_DIR" "$VALUE_FILTERED_OUTPUT_SQL_FILE" "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE"
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Value filter result failed!"
        exit;
    fi
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
    python3 -m scripts.value_postprocessing.value_matching_evaluate "$TABLES_FILE" "$DB_DIR" "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Value filter evaluation failed!"
        exit;
    fi
    echo "Value filter evaluation complete!"
    echo "=================================================================="
else
    echo "Value filter result exist!"
    echo "=================================================================="
fi


# Final Evaluation
if [ -f $VALUE_FILTERED_OUTPUT_SQL_FILE -a ! -f $EVALUATE_OUTPUT_FILE ]; then
    echo "Start evaluate"
    python3 -m utils.evaluation.evaluate --gold "$GOLD_SQL_FILE" --pred "$VALUE_FILTERED_OUTPUT_SQL_FILE" --etype "match" --db "$DB_DIR" --table "$TABLES_FILE" --candidates "$VALUE_FILTERED_OUTPUT_TOPK_SQL_FILE" > "$EVALUATE_OUTPUT_FILE"
    echo "Evaluation Finished!"
    echo "Evaluation result saved in $EVALUATE_OUTPUT_FILE"
    echo "=================================================================="
else
    echo "Evaluation result exist!"
    echo "=================================================================="
fi

echo "Testing Pipeline completed!"