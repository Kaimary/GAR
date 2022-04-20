GAR: A Generate-and-Rank Approach for Natural Language to SQL Translation

# Install & Configure
1. Install the rest of required packages
> pip install -r requirements.txt
2. Download the Spider/GEO datasets

# Training
## GAR is trained with the following template:
> bash train\_pipeline.sh "spider" <train_data_path> <dev_data_path> <table_path> <db_dir> <serialization_num> 0 "nli-distilroberta-base-v2" "roberta-base" "bertpooler"
