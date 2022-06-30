import json
from typing import List
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.models import Model
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register("sql2text_predictor")
class TextClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.query = ''
        self.question = ''
        

    @overrides
    def load_line(self, line: str) -> JsonDict:
        query, question, db_id = line.strip().split("\t")
        self.db_id = db_id
        self.query = query
        self.question = question

        return {"db_id": db_id, "query": query, "question": question}

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs, indent=4) + "\n"
    
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        return outputs["predicted_text"]

    def predict(self, sql: str) -> JsonDict:
        return self.predict_json({"query": sql})

    def predict_batch(self, sqls: List[str]) -> JsonDict:
        inputs = [{"query": sql} for sql in sqls]
        return self.predict_batch_json(inputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        # return self._dataset_reader.text_to_instance(json_dict['db_id'], json_dict['query'])
        return self._dataset_reader.text_to_instance(json_dict['query'])


