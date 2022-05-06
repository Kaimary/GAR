GAR: A Generate-and-Rank Approach for Natural Language to SQL Translation

# Install & Configure
1. Install the required packages
> pip install -r requirements.txt
2. Create <strong>datasets</strong> folder.
3. Download the [Spider](https://yale-lily.github.io/spider) and [GEO](https://github.com/sriniiyer/nl2sql/tree/master/data/geo) datasets, and put the data into the <strong>datasets</strong> folder.

# Training
## GAR can be trained with the following command:
> bash train\_pipeline.sh <dataset_name> <train_data_path> <dev_data_path> <table_path> <db_dir> <serialization_num> <is_overwrite_serialization> "nli-distilroberta-base-v2" "roberta-base" "bertpooler"

The training includes four phasees:
1. Retrieval model training data generation. <em>Please note that this phase expects to take some time to generate a large set of SQL-dialect pairs for each training databases.</em>
2. Retrieval model training
3. Re-ranking model training data generation
4. Re-ranking model training

# Inference
> bash test\_pipeline.sh <dataset_name>  <test_file_path> <table_path> <db_dir> <serialization_num> <rerank_candidate_num> <is_overwrite_serialization> "stsb-mpnet-base-v2" "roberta-base" "bertpooler" <gold_sql_file> "False"

# License
This project is licensed under the GPL-3.0 license.