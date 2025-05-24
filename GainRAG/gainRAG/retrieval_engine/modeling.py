import os
import copy
import torch
from transformers import BertModel, AutoConfig
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)

EMBEDDINGS_DIM: int = 768


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0).clone()
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1).clone() / attention_mask.sum(dim=1)[..., None].clone()
        elif self.config.pooling == "sqrt":
            emb = last_hidden.sum(dim=1) / torch.sqrt(attention_mask.sum(dim=1)[..., None].float())
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1).clone()
        return emb