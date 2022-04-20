local DATA_ROOT = "sqlgenv2/output/spider/reranker/";
local MODEL_NAME = "bert-base-uncased";
{
  "dataset_reader": {
    "type": "listwise_pair_ranker_reader",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": MODEL_NAME,
      "add_special_tokens": false
    },
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME
      }
    },
    "max_instances": 100000
  },
  "validation_dataset_reader": {
    "type": "listwise_pair_ranker_reader",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": MODEL_NAME,
      "add_special_tokens": false
    },
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME
      }
    }
  },
  "train_data_path": DATA_ROOT + "spider_reranker_train.json",
  "validation_data_path": DATA_ROOT + "spider_reranker_dev.json",
  "model": {
    "type": "listwise_pair_ranker",
    "dropout": 0.2,
    "text_field_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer",
          "model_name": MODEL_NAME
        }
      }
    },
    "encoder": {
<<<<<<< HEAD
      "embedding_dim": 768,
      "num_filters": 100,
      "ngram_filter_sizes": [2,3,4,5]
=======
      "pretrained_model": "roberta-base",
      "type": "bert_pooler"
    },
    "encoder_1": {
      "embedding_dim": 768,
      "num_filters": 96,
    },
    "encoder_2": {
      "embedding_dim": 768,
      "num_filters": 96,
>>>>>>> upstream/main
    }
  },
  "data_loader": {
    "type": "pytorch_dataloader",
    "batch_size" : 10 
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "validation_metric": "+ndcg",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam", // "huggingface_adamw",
      "lr": 0.5e-05
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    }
  }
}
