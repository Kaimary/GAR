#!/bin/bash

[ -z "$1" ] && echo "First argument should be the name of the dataset." && exit 1
DATASET_NAME="$1"

[ -z "$2" ] && echo "Second argument should be the dataset train file." && exit 1
DATASET_TRAIN_FILE="$2"

[ -z "$3" ] && echo "Third argument should be the dataset dev file." && exit 1
DATASET_DEV_FILE="$3"

[ -z "$4" ] && echo "Fourth argument should be the schema file of the dataset." && exit 1
TABLES_FILE="$4"

[ -z "$5" ] && echo "Fifth argument should be the directory of the databases for the dataset." && exit 1
DB_DIR="$5"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DIR=$(pwd)

while true; do
    echo "Dataset name: $DATASET_NAME"
    echo "Dataset file(Train): $DATASET_TRAIN_FILE"
    echo "Dataset file(Validation): $DATASET_DEV_FILE"
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
echo "ACTION REPORT: Training pipeline starts ......"
output=`python3 -m configs.get_config_for_bash $DATASET_NAME`
RETRIEVAL_MODEL_TRAIN_DATA_GZ_FILE=$(cut -d'@' -f1 <<< "$output")
RETRIEVAL_MODEL_DIR=$(cut -d'@' -f2 <<< "$output")
RERANKER_TRAIN_DATA_FILE=$(cut -d'@' -f3 <<< "$output")
RERANKER_DEV_DATA_FILE=$(cut -d'@' -f4 <<< "$output")
RERANKER_CONFIG_FILE=$(cut -d'@' -f5 <<< "$output")
RERANKER_MODEL_DIR=$(cut -d'@' -f6 <<< "$output")
RETRIEVAL_MODEL_NAME=$(cut -d'@' -f7 <<< "$output")
RERANKER_MODEL_NAME=$(cut -d'@' -f8 <<< "$output")
RERANKER_EMBEDDING_MODEL_NAME=$(cut -d'@' -f9 <<< "$output")

RETRIEVAL_MODEL_DIR=$RETRIEVAL_MODEL_DIR/$RETRIEVAL_MODEL_NAME
RERANKER_MODEL_DIR=$RERANKER_MODEL_DIR/$RERANKER_MODEL_NAME\+$RERANKER_EMBEDDING_MODEL_NAME

if [ ! -f $RETRIEVAL_MODEL_TRAIN_DATA_GZ_FILE ]; then
    echo "ACTION REPORT: Generate retrieval model's fine-tune data ......"
    python3 -m scripts.retrieval_model_train_script $DATASET_NAME $DATASET_TRAIN_FILE \
    $DATASET_DEV_FILE $TABLES_FILE $DB_DIR $RETRIEVAL_MODEL_DIR
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Retrieval model fine-tune data failed!"
        exit;
    fi
    echo "RESULT REPORT: Retrieval model fine-tune data is ready now!"
    echo "=================================================================="
else
    echo "Retrieval model fine-tune data exists!"
    echo "=================================================================="
fi

if [ ! -d $RETRIEVAL_MODEL_DIR ]; then
    echo "ACTION REPORT: Start to fine-tune the retrieval model ......"
    python3 -m ranker.sentenceBERT.sentence_embedder $DATASET_NAME $RETRIEVAL_MODEL_NAME $RETRIEVAL_MODEL_DIR
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Retrieval model fine-tune failed!"
        exit;
    fi
    echo "RESULT REPORT: Retrieval model fine-tune complete!"
    echo "=================================================================="
else
    echo "Retrieval model model exists!"
    echo "=================================================================="
fi

if [ ! -f $RERANKER_TRAIN_DATA_FILE ]; then
    echo "ACTION REPORT: Generate re-ranker train data ......"
    python3 -m scripts.reranker_script $DATASET_NAME $DATASET_TRAIN_FILE $TABLES_FILE $DB_DIR $RERANKER_MODEL_DIR "train"
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Re-ranker train data failed!"
        exit;
    fi
    echo "RESULT REPORT: Re-ranker train data is ready now!"
    echo "=================================================================="
else
    echo "Re-ranker train data exists!"
    echo "=================================================================="
fi
if [ ! -f $RERANKER_DEV_DATA_FILE ]; then
    echo "ACTION REPORT: Generate re-ranker dev data ......"
    python3 -m scripts.reranker_script $DATASET_NAME $DATASET_DEV_FILE $TABLES_FILE $DB_DIR $RERANKER_MODEL_DIR "dev"
    if [ $? -ne 0 ]; then
        echo "RESULT REPORT: Re-ranker dev data failed!"
        exit;
    fi
    echo "RESULT REPORT: Re-ranker dev data is ready now!"
    echo "=================================================================="
else
    echo "Re-ranker dev data exists!"
    echo "=================================================================="
fi

if [ ! -d $RERANKER_MODEL_DIR ]; then
    echo "=================================================================="
    echo "ACTION REPORT: Change the embedding model name in the config file..."
    python3 -m configs.update_config "$RERANKER_CONFIG_FILE" "$RERANKER_EMBEDDING_MODEL_NAME" "$RERANKER_TRAIN_DATA_FILE" "$RERANKER_DEV_DATA_FILE"
    echo "RESULT REPORT: Re-ranker model config file update complete!"
    echo "ACTION REPORT: Start to train re-ranker model ......"
    allennlp train "$RERANKER_CONFIG_FILE" -s "$RERANKER_MODEL_DIR" --include-package ranker.BERTPooler.dataset_readers.listwise_pair_reader --include-package ranker.BERTPooler.models.listwise_pair_ranker || exit $?
    echo "RESULT REPORT: Re-ranker model training complete!"
else
    echo "Re-ranker model exists!"
    echo "=================================================================="
fi
echo "Training Pipeline completed!"
