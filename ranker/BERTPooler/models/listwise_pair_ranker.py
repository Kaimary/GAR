from typing import Dict
from overrides import overrides
import torch
from torch import nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Auc

# from ranker.common.losses.listNet import listNet_loss
# from ranker.common.losses.binary_listNet import binary_listNet_loss
# from ranker.common.losses.listMLE import listMLE
from ranker.common.losses.neuralNDCG import neuralNDCG
from ranker.common.metrics import NDCG, MRR


@Model.register("listwise_pair_ranker")
class ListwisePairRanker(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            dropout: float = None,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._dropout = dropout and torch.nn.Dropout(dropout)
        self._encoder = encoder
        self._denseLayer = nn.Linear(self._encoder.get_output_dim(), 1, bias=False)

        self._auc = Auc()
        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)

        initializer(self)

    # @torchsnooper.snoop()
    def forward(  # type: ignore
            self,
            query_dialect_pairs: TextFieldTensors,  # batch * num_dialects * words
            labels: torch.IntTensor = None  # batch * num_labels
    ) -> Dict[str, torch.Tensor]:

        embedded_query_dialect_pairs = self._text_field_embedder(query_dialect_pairs,
                                                                 num_wrapping_dims=1)  # options_mask.dim() - 2
        query_dialect_pairs_mask = get_text_field_mask(query_dialect_pairs).long()

        batch_size, num_pairs, num_tokens, _ = embedded_query_dialect_pairs.size()

        if self._dropout:
            embedded_query_dialect_pairs = self._dropout(embedded_query_dialect_pairs)

        encoder_outputs = self._encoder(
            embedded_query_dialect_pairs.view(batch_size * num_pairs, num_tokens, -1),
            query_dialect_pairs_mask.view(batch_size * num_pairs, num_tokens)
        )
        encoder_outputs = encoder_outputs.view(batch_size, num_pairs, -1)
        scores = self._denseLayer(encoder_outputs).squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "probs": probs}
        # output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if labels is not None:
            label_mask = (labels != -1)
            self._mrr(probs, labels, label_mask)
            self._ndcg(scores, labels, label_mask)

            # loss = binary_listNet_loss(scores, labels)
            loss = neuralNDCG(scores, labels)
            probs = probs.view(-1)
            labels = labels.view(-1)
            label_mask = label_mask.view(-1)
            self._auc(probs, labels.ge(0.5).long(), label_mask)

            output_dict["loss"] = loss  # loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()

        return output_dict

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "auc": self._auc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
        }
        return metrics

    default_predictor = "document_ranker"
