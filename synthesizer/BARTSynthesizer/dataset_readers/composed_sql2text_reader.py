from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp_models.generation.dataset_readers import CNNDailyMailDatasetReader
from allennlp.data.fields import TextField, MetadataField


@DatasetReader.register("composed_sql2text_reader")
class ComposedSQL2TextDatasetReader(CNNDailyMailDatasetReader):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                if len(line.strip().split("\t")) > 3:
                    print(line)
                query, question, _ = line.strip().split("\t")
                yield self.text_to_instance(query, question)
    
    def text_to_instance(
        self, source_sequence: str, target_sequence: str = None
    ) -> Instance:  # type: ignore
        # meta_fields = {"db_id": db_id}
        if self._source_prefix is not None:
            tokenized_source = self._source_tokenizer.tokenize(
                self._source_prefix + source_sequence
            )
        else:
            tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            tokenized_source = tokenized_source[: self._source_max_tokens]

        source_field = TextField(tokenized_source)
        if target_sequence is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_sequence)
            if (
                self._target_max_tokens is not None
                and len(tokenized_target) > self._target_max_tokens
            ):
                tokenized_target = tokenized_target[: self._target_max_tokens]
            target_field = TextField(tokenized_target)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
            # return Instance({"source_tokens": source_field, "target_tokens": target_field, "metadata": MetadataField(meta_fields)})
        else:
            return Instance({"source_tokens": source_field})
            # return Instance({"source_tokens": source_field, "metadata": MetadataField(meta_fields)})

