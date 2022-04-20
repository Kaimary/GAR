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

[ -z "$6" ] && echo "Sixth argument should be the number of candidate SQL set that needs to be generated." && exit 1
NUM_SQL="$6"

[ -z "$7" ] && echo "Seventh argument should be the overwrite flag. " && exit 1
OVERWRITE="$7"

[ -z "$8" ] && echo "Eighth argument should be candidate base word embedding model name. " && exit 1
CANDIDATE_EMBEDDING_MODEL_NAME="$8"

[ -z "$9" ] && echo "Ninth argument should be reranking base word embedding model name. " && exit 1
RERANKER_EMBEDDING_MODEL_NAME="$9"

[ -z "${10}" ] && echo "Tenth argument should be reranking model name. " && exit 1
RERANKER_MODEL_NAME="${10}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DIR=$(pwd)

while true; do
    echo "Dataset name: $DATASET_NAME"
    echo "Dataset file(Train): $DATASET_TRAIN_FILE"
    echo "Dataset file(Validation): $DATASET_DEV_FILE"
    echo "Schema file of the dataset: $TABLES_FILE"
    echo "Databases directory of the dataset: $DB_DIR"
    echo "Number of candidate SQLs that needs to be generated: $NUM_SQL"
    echo "Overwrite flag for the generated data(True only if the SQL generator has changed): $OVERWRITE"
    echo "Candidate base word embedding model name(e.g. nli-distilroberta-base-v2): $CANDIDATE_EMBEDDING_MODEL_NAME"
    echo "Reranking base word embedding model name(e.g. bert-base-uncased): $RERANKER_EMBEDDING_MODEL_NAME"
    echo "Reranking model name(e.g. bertpooler): $RERANKER_MODEL_NAME"
    read -p "Is this ok [y/n] ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit 0;;
        * ) echo "Please answer y or n.";;
    esac
done

echo "=================================================================="
echo "ACTION REPORT: Training pipeline starts ......"
output=`python3 -m get_config_for_bash $DATASET_NAME`
SEMSIMILARITY_TRIPLE_DATA_GZ_FILE=$(cut -d'@' -f1 <<< "$output")
SENTENCE_EMBEDDER_MODEL_DIR=$(cut -d'@' -f2 <<< "$output")
RERANKER_TRAIN_DATA_FILE=$(cut -d'@' -f3 <<< "$output")
RERANKER_DEV_DATA_FILE=$(cut -d'@' -f4 <<< "$output")
RERANKER_CONFIG_FILE=$(cut -d'@' -f5 <<< "$output")
RERANKER_MODEL_DIR=$(cut -d'@' -f6 <<< "$output")
SENTENCE_EMBEDDER_MODEL_DIR=$SENTENCE_EMBEDDER_MODEL_DIR\_$CANDIDATE_EMBEDDING_MODEL_NAME
RERANKER_MODEL_DIR=$RERANKER_MODEL_DIR\_$RERANKER_MODEL_NAME\_$RERANKER_EMBEDDING_MODEL_NAME
if [ ! -f $SEMSIMILARITY_TRIPLE_DATA_GZ_FILE ]; then
    echo "ACTION REPORT: Generate sentence embedder fine-tune data ......"
    python3 -m datagen.model_datagen.semsimilarity_triples_data_gen $DATASET_NAME $DATASET_TRAIN_FILE $DATASET_DEV_FILE $TABLES_FILE $DB_DIR $NUM_SQL $OVERWRITE
    echo "RESULT REPORT: Sentence embedder fine-tune data is ready now!"
    echo "=================================================================="
else
    echo "Sentence embedder fine-tune data exists!"
    echo "=================================================================="
fi
if [ ! -d $SENTENCE_EMBEDDER_MODEL_DIR ]; then
    echo "ACTION REPORT: Start to fine-tune the sentence embedder ......"
    python3 -m models.semantic_matcher.sentence_transformers.sentence_embedder $CANDIDATE_EMBEDDING_MODEL_NAME $SENTENCE_EMBEDDER_MODEL_DIR $DATASET_NAME
    echo "RESULT REPORT: Sentence embedder fine-tune complete!"
    echo "=================================================================="
else
    echo "Sentence embedder model exists!"
    echo "=================================================================="
fi
if [ ! -f $RERANKER_TRAIN_DATA_FILE ]; then
    echo "ACTION REPORT: Generate re-ranker train/dev data ......"
    python3 -m datagen.model_datagen.re-ranker_listwise_data_gen_for_100 $DATASET_NAME $DATASET_TRAIN_FILE $DATASET_DEV_FILE $TABLES_FILE $DB_DIR $NUM_SQL $OVERWRITE $SENTENCE_EMBEDDER_MODEL_DIR
    echo "RESULT REPORT: Re-ranker train/dev data is ready now!"
    echo "=================================================================="
else
    echo "Re-ranker train/dev data exists!"
    echo "=================================================================="
fi
if [ ! -d $RERANKER_MODEL_DIR ]; then
    echo "=================================================================="
    echo "ACTION REPORT: Change the embedding model name in the config file..."
    python3 -m update_config "$RERANKER_CONFIG_FILE" "$RERANKER_EMBEDDING_MODEL_NAME" "$RERANKER_TRAIN_DATA_FILE" "$RERANKER_DEV_DATA_FILE"
    echo "RESULT REPORT: Re-ranker model config file update complete!"
    echo "ACTION REPORT: Start to train re-ranker model ......"
    allennlp train "$RERANKER_CONFIG_FILE" -s "$RERANKER_MODEL_DIR" --include-package dataset_readers.listwise_pair_reader --include-package models.semantic_matcher.listwise_pair_ranker || exit $?
    echo "RESULT REPORT: Re-ranker model training complete!"
else
    echo "Re-ranker model exists!"
    echo "=================================================================="
fi
echo "Training Pipeline completed!"
