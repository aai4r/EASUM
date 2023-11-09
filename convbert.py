import logging
import random
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import GELUActivation
from transformers.models.convbert.modeling_convbert import ConvBertPreTrainedModel
from transformers.models.convbert.modeling_convbert import ConvBertEmbeddings, ConvBertEncoder
from modeling import MAG
from global_configs import *

logger = logging.getLogger(__name__)

ACT2FN = {
    "gelu": GELUActivation()
}


class ConvBertClassificationHead_emo(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        config.num_labels = 10
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, hidden_states: torch.Tensor, DG=False, **kwargs) -> torch.Tensor:
        if not DG:
            x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = hidden_states
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvBertClassificationHead_emo_6(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        config.num_labels = 6
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, hidden_states: torch.Tensor, DG=False, **kwargs) -> torch.Tensor:
        if not DG:
            x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = hidden_states
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvBertClassificationHead_sent(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        config.num_labels = 1
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, hidden_states: torch.Tensor, DG=False, **kwargs) -> torch.Tensor:
        if not DG:
            x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = hidden_states
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvBertModel(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvBertEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = ConvBertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @staticmethod
    def augment(aug, embedding):
        N = (aug[aug == True].shape)[0]
        if N != 0:
            temp = embedding[aug == True]
            for i, data in enumerate(embedding[aug == True]):
                zero_out = random.sample(list(range(embedding.shape[1])), 5)
                temp[i][zero_out] = 0
            embedding[aug == True] = temp
            return embedding
        else:
            return embedding

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            aug=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if aug is not None:
            hidden_states = self.augment(aug, hidden_states)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


class ConvBertForSequenceClassification(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.convbert = ConvBertModel(config)
        self.classifier_sent = ConvBertClassificationHead_sent(config)
        self.classifier_emo = ConvBertClassificationHead_emo(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            dataset=None,
            aug=None,
            reverse=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # NOTE: visual, acoustic term is added!
        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aug=aug
        )

        sequence_output = outputs[0]
        if dataset == 'mosi' or dataset == 'mosei':
            if not reverse:
                logits = self.classifier_sent(sequence_output)
            else:
                logits = self.classifier_emo(sequence_output)
        else:
            if not reverse:
                logits = self.classifier_emo(sequence_output)
            else:
                logits = self.classifier_sent(sequence_output)

        return logits, sequence_output[:, 0, :]


class MAG_ConvBertModel(ConvBertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvBertEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = ConvBertEncoder(config)

        self.MAG = MAG(
            config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @staticmethod
    def augment(aug, text, audio, video):
        """
        text: temporal zero out
        audio & video: random noise
        """
        N = (aug[aug == True].shape)[0]
        if N != 0:
            for i in range(N):
                # alpha_text = torch.normal(mean=1, std=torch.eye(TEXT_SEQ_LEN, TEXT_DIM))
                # beta_text = torch.normal(mean=0, std=torch.eye(TEXT_SEQ_LEN, TEXT_DIM))
                alpha_acoustic = torch.normal(mean=1, std=torch.eye(ACOUSTIC_SEQ_LEN, ACOUSTIC_DIM))
                beta_acoustic = torch.normal(mean=0, std=torch.eye(ACOUSTIC_SEQ_LEN, ACOUSTIC_DIM))
                alpha_visual = torch.normal(mean=1, std=torch.eye(VISUAL_SEQ_LEN, VISUAL_DIM))
                beta_visual = torch.normal(mean=0, std=torch.eye(VISUAL_SEQ_LEN, VISUAL_DIM))
                if i == 0:
                    # alpha_text_n = alpha_text.unsqueeze(0)
                    # beta_text_n = beta_text.unsqueeze(0)
                    alpha_acoustic_n = alpha_acoustic.unsqueeze(0)
                    beta_acoustic_n = beta_acoustic.unsqueeze(0)
                    alpha_visual_n = alpha_visual.unsqueeze(0)
                    beta_visual_n = beta_visual.unsqueeze(0)
                else:
                    # alpha_text_n = torch.cat((alpha_text_n, alpha_text.unsqueeze(0)), dim=0)
                    # beta_text_n = torch.cat((beta_text_n, beta_text.unsqueeze(0)), dim=0)
                    alpha_acoustic_n = torch.cat((alpha_acoustic_n, alpha_acoustic.unsqueeze(0)), dim=0)
                    beta_acoustic_n = torch.cat((beta_acoustic_n, beta_acoustic.unsqueeze(0)), dim=0)
                    alpha_visual_n = torch.cat((alpha_visual_n, alpha_visual.unsqueeze(0)), dim=0)
                    beta_visual_n = torch.cat((beta_visual_n, beta_visual.unsqueeze(0)), dim=0)

            # alpha_text_n = alpha_text_n.to(DEVICE)
            # beta_text_n = beta_text_n.to(DEVICE)
            alpha_acoustic_n = alpha_acoustic_n.to(DEVICE)
            beta_acoustic_n = beta_acoustic_n.to(DEVICE)
            alpha_visual_n = alpha_visual_n.to(DEVICE)
            beta_visual_n = beta_visual_n.to(DEVICE)

            # text[aug == True] = alpha_text_n * text[aug == True] + beta_text_n
            video[aug == True] = alpha_visual_n * video[aug == True] + beta_visual_n
            audio[aug == True] = alpha_acoustic_n * audio[aug == True] + beta_acoustic_n

            temp = text[aug == True]
            for i, data in enumerate(text[aug == True]):
                zero_out = random.sample(list(range(text.shape[1])), 5)
                temp[i][zero_out] = 0
            text[aug == True] = temp
            return text, audio, video
        else:
            return text, audio, video

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            aug=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if aug is not None:
            hidden_states, acoustic, visual = self.augment(aug, hidden_states, acoustic, visual)

        # Early fusion with MAG (important!)
        fused_embedding = self.MAG(hidden_states, visual, acoustic)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(fused_embedding)

        hidden_states = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


################################################################################################


class MAG_ConvBertForSequenceClassification_DG(ConvBertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.convbert = MAG_ConvBertModel(config, multimodal_config)
        self.classifier_sent = ConvBertClassificationHead_sent(config)
        self.classifier_emo = ConvBertClassificationHead_emo(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids,
            acoustic,
            visual,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            dataset=None,
            aug=None,
            reverse=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # NOTE: visual, acoustic term is added!
        outputs = self.convbert(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aug=aug
        )

        sequence_output = outputs[0]

        logits, aux_logits = None, None
        if dataset == 'mosi' or dataset == 'mosei':
            if not reverse:
                logits = self.classifier_sent(sequence_output)
            else:
                aux_logits = self.classifier_emo(sequence_output)
        else:
            if not reverse:
                logits = self.classifier_emo(sequence_output)
            else:
                aux_logits = self.classifier_sent(sequence_output)

        return logits, aux_logits, sequence_output[:, 0, :]


class MAG_ConvBertForSequenceClassification_DS(ConvBertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.convbert = MAG_ConvBertModel(config, multimodal_config)
        self.classifier_sent = ConvBertClassificationHead_sent(config)
        # self.classifier_emo = ConvBertClassificationHead_emo(config)
        self.classifier_emo_6 = ConvBertClassificationHead_emo_6(config)
        self.fusion_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids,
            acoustic,
            visual,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            dataset=None,
            aug=None,
            DG_feat=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # NOTE: visual, acoustic term is added!
        outputs = self.convbert(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aug=aug
        )

        sequence_output = outputs[0]
        if DG_feat is not None:
            sequence_output = F.relu(self.fusion_layer(torch.cat([sequence_output[:, 0, :], DG_feat], dim=1)), inplace=False)

        if dataset == 'mosi' or dataset == 'mosei':
            logits = self.classifier_sent(sequence_output, DG=True)
            aux_logits = self.classifier_emo(sequence_output, DG=True)
        else:
            logits = self.classifier_emo_6(sequence_output, DG=True)
            aux_logits = self.classifier_sent(sequence_output, DG=True)

        return logits, aux_logits, sequence_output

