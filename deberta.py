import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaEmbeddings, DebertaEncoder, ContextPooler, StableDropout

from modeling import MAG
from global_configs import *


class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.MAG = MAG(
            self.config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        fused_embedding = self.MAG(embedding_output, visual, acoustic)

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]


class MAG_DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 1)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config, multimodal_config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.fusion_layer = nn.Linear(output_dim + 768, output_dim)

        self.classifier_sent = nn.Linear(output_dim, 1)
        self.classifier_emo = nn.Linear(output_dim, 10)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids,
        acoustic,
        visual,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        dataset=None,
        DG_feat=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.deberta(
            input_ids,
            visual,
            acoustic,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        if DG_feat is not None:
            pooled_output = F.relu(self.fusion_layer(torch.cat([pooled_output, DG_feat], dim=1)), inplace=False)

        pooled_output = self.dropout(pooled_output)
        if dataset == 'mosi' or dataset == 'mosei':
            logits = self.classifier_sent(pooled_output)
            aux_logits = self.classifier_emo(pooled_output)
            return logits, aux_logits, None
        else:
            logits = self.classifier_emo(pooled_output)
            log_prob = F.log_softmax(logits, 1)
            aux_logits = self.classifier_sent(pooled_output)
            return log_prob, aux_logits, None

        # loss = None
        # logits = None
        # if labels is not None:
        #     if self.num_labels == 1:
        #         # regression task
        #         loss_fn = nn.MSELoss()
        #         logits = logits.view(-1).to(labels.dtype)
        #         loss = loss_fn(logits, labels.view(-1))
        #     elif labels.dim() == 1 or labels.size(-1) == 1:
        #         label_index = (labels >= 0).nonzero()
        #         labels = labels.long()
        #         if label_index.size(0) > 0:
        #             labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
        #             labels = torch.gather(labels, 0, label_index.view(-1))
        #             loss_fct = CrossEntropyLoss()
        #             loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
        #         else:
        #             loss = torch.tensor(0).to(logits)
        #     else:
        #         log_softmax = nn.LogSoftmax(-1)
        #         loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        #
        # output = (logits,) + outputs[1:]
        # return ((loss,) + output) if loss is not None else output
