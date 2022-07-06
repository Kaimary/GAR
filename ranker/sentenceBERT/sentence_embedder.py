"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import sys
import math
import logging
import os
import gzip
import csv
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from configs.config import DIR_PATH, RETRIEVAL_MODEL_TRAIN_DATA_GZ_FILE, RETRIEVAL_MODEL_DIR, \
    RETRIEVAL_MODEL_TRAIN_BATCH_SIZE, RETRIVIAL_MODEL_LEARNING_RATE

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

sts_dataset_path = DIR_PATH + RETRIEVAL_MODEL_TRAIN_DATA_GZ_FILE.format(sys.argv[1])
if not os.path.exists(sts_dataset_path):
    print(f"No data found! Please generate the data firstly.")

dataset_name = sys.argv[1]
model_name = sys.argv[2]
num_epochs = 10
train_batch_size = RETRIEVAL_MODEL_TRAIN_BATCH_SIZE 
model_save_path = os.path.join(DIR_PATH + RETRIEVAL_MODEL_DIR.format(dataset_name), model_name)

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read training data triples...")
train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Read validation data triples...")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='spider-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': RETRIVIAL_MODEL_LEARNING_RATE},
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on Spider benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='spider-test')
test_evaluator(model, output_path=model_save_path)
