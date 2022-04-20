local DATA_ROOT = "datasets/spider/ranker/";

{
  "dataset_reader": {
    "type": "listwise_ranker_reader",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "bert-base-uncased"
    },
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer",
        "model_name": "bert-base-uncased"
      }
    },
    "lazy": true,
    "max_instances": 100000
  },
  "train_data_path": DATA_ROOT + "train3.json",
  "validation_data_path": DATA_ROOT + "dev3.json",
  "model": {
    "type": "listwise_ranker",
    "dropout": 0.35,
    "text_field_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer",
          "model_name": "bert-base-uncased"
        }
      }
    },
    "relevance_matcher": {
      "pretrained_model": "bert-base-uncased",
      "input_dim": 768,
      "type": "bert_cls" 
    }
  },
  "data_loader": {
    "type": "pytorch_dataloader",
    "batch_size" : 15 
  },
  "validation_data_loader": {
    "type": "pytorch_dataloader",
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "validation_metric": "+ndcg",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam", // "huggingface_adamw",
      "lr": 1e-05
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    }
  }
}
