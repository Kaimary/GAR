{
    "data_loader": {
        "batch_size": 10,
        "type": "pytorch_dataloader"
    },
    "dataset_reader": {
        "lazy": true,
        "max_instances": 100000,
        "token_indexers": {
            "bert": {
                "model_name": "roberta-base",
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            "add_special_tokens": false,
            "model_name": "roberta-base",
            "type": "pretrained_transformer"
        },
        "type": "listwise_pair_ranker_reader"
    },
    "model": {
        "dropout": 0.2,
        "encoder": {
            "pretrained_model": "roberta-base",
            "type": "bert_pooler"
        },
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "model_name": "roberta-base",
                    "type": "pretrained_transformer"
                }
            }
        },
        "type": "listwise_pair_ranker"
    },
    "train_data_path": "/home/kaimary/GAR/saved_data/spider/reranker/train.json",
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "factor": 0.5,
            "mode": "max",
            "patience": 0,
            "type": "reduce_on_plateau"
        },
        "num_epochs": 50,
        "optimizer": {
            "lr": 5e-06,
            "type": "adam"
        },
        "patience": 10,
        "validation_metric": "+ndcg"
    },
    "validation_data_path": "/home/kaimary/GAR/saved_data/spider/reranker/dev.json",
    "validation_dataset_reader": {
        "token_indexers": {
            "bert": {
                "model_name": "roberta-base",
                "type": "pretrained_transformer"
            }
        },
        "tokenizer": {
            "add_special_tokens": false,
            "model_name": "roberta-base",
            "type": "pretrained_transformer"
        },
        "type": "listwise_pair_ranker_reader"
    }
}