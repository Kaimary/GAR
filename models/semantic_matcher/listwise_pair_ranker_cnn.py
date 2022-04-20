from typing import Dict
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides
import torch
from torch import nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Auc

from models.losses.listNet import listNet_loss
from models.losses.binary_listNet import binary_listNet_loss
from models.losses.listMLE import listMLE
from models.losses.neuralNDCG import neuralNDCG
from ..metrics import NDCG, MRR

@Model.register("listwise_pair_ranker")
class ListwisePairRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder:Seq2VecEncoder,
        encoder_1: CnnEncoder,
        encoder_2: CnnEncoder,
        dropout: float = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._dropout = dropout and torch.nn.Dropout(dropout)
        # self._encoder_1 = encoder()
        # self._encoder_2 = encoder_2
        self._encoder_1 = encoder_1
        self._encoder_2 = encoder_2
        self._denseLayer = nn.Linear(768, 1, bias=False)
        # self._denseLayer_2 = nn.Linear(self._encoder_2.get_output_dim(), 1, bias=False)
        # self._denseLayer = nn.Linear(self._encoder_1.get_output_dim(), 1, bias=False)

        self._auc = Auc()
        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)

        initializer(self)

    # @torchsnooper.snoop()
    def forward(  # type: ignore
        self, 
        dialects: TextFieldTensors, # batch * num_dialects * words
        querys: TextFieldTensors, # 
        labels: torch.IntTensor = None # batch * num_labels
    ) -> Dict[str, torch.Tensor]:
        # print("dialects['bert']['token_ids'].size() = ", dialects['bert']['token_ids'].size())
        # print("querys['bert']['token_ids'].size() = ", querys['bert']['token_ids'].size())
        embedded_querys = self._text_field_embedder(querys, num_wrapping_dims=1)
        embedded_dialects = self._text_field_embedder(dialects, num_wrapping_dims=1)
        # print("embedded_querys.size() = ", embedded_querys.size())
        # print("embedded_dialects.size() = ", embedded_dialects.size())
        querys_mask = get_text_field_mask(querys).long()
        dialects_mask = get_text_field_mask(dialects).long()
        # print("dialects_mask.size() = ", dialects_mask.size())
        # print("querys_mask.size() = ", querys_mask.size())
        querys_batch_size, querys_num_pairs, querys_num_tokens, _ = embedded_querys.size()
        # print(querys_batch_size, querys_num_pairs, querys_num_tokens)
        dialects_batch_size, dialects_num_pairs, dialects_num_tokens, _ = embedded_dialects.size()
        # print(dialects_batch_size, dialects_num_pairs, dialects_num_tokens)

        if self._dropout:
            embedded_querys = self._dropout(embedded_querys)
            embedded_dialects = self._dropout(embedded_dialects)
        # print("embedded_dialects.size() = ", embedded_dialects.size())
        # print("embedded_querys.size() = ", embedded_querys.size())

        encoder_1_outputs = self._encoder_1(
            embedded_querys.view(querys_batch_size*querys_num_pairs, querys_num_tokens, -1),
            querys_mask.view(querys_batch_size*querys_num_pairs, querys_num_tokens)
        )
        # print("encoder_1_outputs.size() = ", encoder_1_outputs.size())
        encoder_2_outputs = self._encoder_2(
            embedded_dialects.view(dialects_batch_size*dialects_num_pairs, dialects_num_tokens, -1),
            dialects_mask.view(dialects_batch_size*dialects_num_pairs, dialects_num_tokens)
        )
        # print("encoder_2_outputs.size() = ", encoder_2_outputs.size())

        # encoder_outputs = torch.cat((encoder_1_outputs, encoder_2_outputs), 1)
        # print("encoder_outputs.size() = ", encoder_outputs.size())

        # encoder_1_outputs = encoder_1_outputs + encoder_2_outputs
        # print("encoder_1_outputs.size() = ", encoder_1_outputs.size())

        encoder_1_outputs = encoder_1_outputs.view(querys_batch_size, querys_num_pairs , -1)
        # print("encoder_1_outputs.size() = ", encoder_1_outputs.size())
        encoder_2_outputs = encoder_2_outputs.view(querys_batch_size, querys_num_pairs , -1)
        # print("encoder_2_outputs.size() = ", encoder_2_outputs.size())

        encoder_outputs = torch.cat((encoder_1_outputs, encoder_2_outputs), 2)
        # print("encoder_outputs.size() = ", encoder_outputs.size())

        scores = self._denseLayer(encoder_outputs).squeeze(-1)

        # scores_1 = self._denseLayer_1(encoder_1_outputs).squeeze(-1)
        # print("scores_1 = ", scores_1.size())
        # scores_2 = self._denseLayer_2(encoder_2_outputs).squeeze(-1)
        # print("scores_2 = ", scores_2.size())
        # # scores = scores_1 + scores_2
        # scores = torch.cat((scores_1, scores_2), 1)
        # print("scores = ", scores.size())
        probs = torch.sigmoid(scores)
        # print("probs = ", probs.size())

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
            
            output_dict["loss"] = loss #loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()
        return output_dict


        """
        embedded_query_dialect_pairs = self._text_field_embedder(query_dialect_pairs, num_wrapping_dims=2) # options_mask.dim() - 2
        print("embedded_query_dialect_pairs.size() = ", embedded_query_dialect_pairs.size())
        query_dialect_pairs_mask = get_text_field_mask(query_dialect_pairs).long()
        print("query_dialect_pairs_mask.size() = ", query_dialect_pairs_mask.size())
        batch_size, num_pairs, _ , num_tokens, _ = embedded_query_dialect_pairs.size()

        if self._dropout:
            embedded_query_dialect_pairs = self._dropout(embedded_query_dialect_pairs)

        encoder_outputs = self._encoder(
            embedded_query_dialect_pairs.view(batch_size*num_pairs, num_tokens, -1), 
            query_dialect_pairs_mask.view(batch_size*num_pairs, num_tokens)
            )
        encoder_outputs = encoder_outputs.view(batch_size, num_pairs , -1)
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
            
            output_dict["loss"] = loss #loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()
        return output_dict
        """
        

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
