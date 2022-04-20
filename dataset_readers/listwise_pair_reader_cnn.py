from typing import List, Union, Tuple
from overrides import overrides

import json
import numpy as np
import pandas as pd

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, ListField
from allennlp.data.instance import Instance

from dataset_readers.ir_reader import IRDatasetReader

@DatasetReader.register("listwise_pair_ranker_reader")
class ListwisePairRankingReader(IRDatasetReader):

    @overrides
    def _read(self, file_path: str):
        if not file_path.endswith('.json'):
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for ex in json_obj:
                if 'labels' in ex.keys():
                    ins = self.text_to_instance(
                        query=ex['question'],
                        dialects=ex['candidates'],
                        labels=ex['labels'])
                else:
                    ins = self.text_to_instance(
                        query=ex['question'],
                        dialects=ex['candidates'])

                if ins is not None:
                    yield ins
        
    @overrides
    def text_to_instance(
        self,
        query: Union[str, Tuple], 
        dialects: List[str],
        labels: Union[str, float] = None,
        **kwargs
    ) -> Instance:  # type: ignore
        dialects = list(filter(None, dialects))
        
        if labels:
            assert all(l >= 0 for l in labels)
            assert all((l == 0) for l in labels[len(dialects):])
            labels = labels[:len(dialects)]

            
        """
        # solution 1: nl+dia pair
        
        """
        """
        # query_field = self._make_textfield(query)
        query_dialect_pairs_field = ListField([self._make_pair_textfield((query, o)) for o in dialects])
        
        fields = { 
            # 'tokens': query_field, 
            'query_dialect_pairs': query_dialect_pairs_field 
            # 'metadata': MetadataField(kwargs)
            }

        if labels:
            labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        """

        """
        solution 2: 

        """
        """
        query_dialect_pairs_field = ListField([ListField([self._make_textfield(query), self._make_textfield(o)]) for o in dialects])
        fields = {
            'query_dialect_pairs': query_dialect_pairs_field
        }
        if labels:
            labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        """

        """
        solution 3:

        """
        dialect_field  = ListField([self._make_textfield(o) for o in dialects])
        query_field = ListField([self._make_textfield(query) for o in dialects])
        fields = {
            'dialects': dialect_field,
            'querys': query_field,
        }
        if labels:
            labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)

        fields = {k: v for (k, v) in fields.items() if v is not None}


        return Instance(fields)
