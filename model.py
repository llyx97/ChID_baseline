import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from allennlp.nn.util import batched_index_select
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from typing import Optional, Tuple, Union

class BertForChID(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # if config.is_decoder:
        #     logger.warning(
        #         "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #         "bi-directional self-attention."
        #     )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'paris'",
    #     expected_loss=0.88,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output) # (Batch_size, Seq_len, Vocab_size)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() 
            batch_size = prediction_scores.size(0)
            candidate_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(-1, prediction_scores.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)
            candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]) # (Batch_size x 4, num_choices)
            mask_len = int(candidate_indices.size(0)/batch_size)
            candidate_logits = batched_index_select(candidate_prediction_scores, candidate_indices).squeeze(-1).reshape(prediction_scores.shape[0], mask_len, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)
            # mask inside the range mask_len, positions of paraphrase tokens are 1 and the paddings are 0
            idiom_mask = candidate_indices.reshape(batch_size, mask_len, -1)!=0
            candidate_final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-2) * idiom_mask.transpose(-2,-1), dim=-1) # (Batch_size, num_choices)
            cand_lens = idiom_mask.sum(dim=1)
            candidate_final_scores.div_(cand_lens)


            if len(labels.size())>1:       # Computing CE loss over the vacabulary
                vocab_size = candidate_prediction_scores.size(1)
                weights = torch.ones(vocab_size).to(candidate_prediction_scores.device)
                weights[0] = 0
                loss_fct = CrossEntropyLoss(weights)
                labels = labels.reshape(-1)
                masked_lm_loss = loss_fct(candidate_prediction_scores.squeeze(-1), labels)
            else:                      # Computing CE loss over the candidates
                #candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]) # (Batch_size x 4, num_choices)
                #candidate_logits = batched_index_select(candidate_prediction_scores, candidate_indices).squeeze(-1).reshape(prediction_scores.shape[0], 4, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)

                candidate_labels = labels.reshape(labels.shape[0], 1).repeat(1, mask_len) # (Batch_size, 4)
                #candidate_final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-2), dim=-1) # (Batch_size, num_choices)

                masked_lm_loss = loss_fct(candidate_logits, candidate_labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=candidate_final_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
