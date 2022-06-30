import json
import os

from synthesizer import abstract_synthesizer
from .dataset_readers.composed_sql2text_reader import *
from .predictors.sql2text_predictor import TextClassifierPredictor
from allennlp.common.params import Params
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive

class BARTSynthesizer(abstract_synthesizer.AbstractNLSynthesizer):
    def __init__(self, config_file, serialization_dir):
        self.config_file = config_file
        self.serialization_dir = serialization_dir
        self.load_model()

    def load_model(self):
        # serialization_dir = "allenmodels/bart_model"
        # config = "./saved_model/config.json"
        # Unarchive from the file
        archive = load_archive(os.path.join(self.serialization_dir, "model.tar.gz"), cuda_device=0)
        print("load archive complete!")
        # loaded_params = Params.from_file(config)
        # loaded_model = Model.load(loaded_params, serialization_dir, 'allenmodels/bart_model/best.th')

        config_json = json.load(open(self.config_file, 'r'))
        params = Params(config_json)
        if "dataset_reader" in params:
            dataset_reader = DatasetReader.from_params(params.pop("dataset_reader"))
        self.predictor = TextClassifierPredictor(archive.model, dataset_reader)

    def synthesize(self, input, batch=False):
        if batch:
            output = self.predictor.predict_batch(input)
            return [o['predicted_text'] for o in output]
