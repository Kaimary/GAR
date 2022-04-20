{
    "data_loader": {
        "batch_size": 2,
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
        "type": "from_archive",
        "archive_file": "sqlgenv2/output/spider/reranker/reranker_bertpooler_roberta-base/model.tar.gz"
    },
    "train_data_path": "sqlgenv2/output/spider/reranker/spider_reranker_train_finetune.json",
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "factor": 0.5,
            "mode": "max",
            "patience": 0,
            "type": "reduce_on_plateau"
        },
        "num_epochs": 10,
        "optimizer": {
            "lr": 5e-06,
            "type": "adam"
        },
        "validation_metric": "+ndcg"
    },
    "validation_data_path": "sqlgenv2/output/spider/reranker/spider_reranker_dev_finetune.json",
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