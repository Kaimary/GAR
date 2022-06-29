from typing import Dict, List, Union, Tuple
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.common.checks import ConfigurationError

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


class IRDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens


    def _make_textfield(self, text: Union[str, Tuple]):
        if not text:
            return None
        
        if isinstance(text, tuple) and not isinstance(self.tokenizer, PretrainedTransformerTokenizer):
            text = ' '.join(text)
        
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    def _make_pair_textfield(self, text: Union[str, Tuple]):
        if not text:
            return None
        
        if isinstance(text, tuple):
            sentence_a, sentence_b = text
            tokens_a = self.tokenizer.tokenize(sentence_a)
            tokens_b = self.tokenizer.tokenize(sentence_b)
            concat_tokens = self.tokenizer.add_special_tokens(tokens_a, tokens_b)
            if self.max_tokens:
                concat_tokens = concat_tokens[:self.max_tokens]
            
        return TextField(concat_tokens, token_indexers=self.token_indexers)