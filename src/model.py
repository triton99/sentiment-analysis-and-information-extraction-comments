import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from transformers import RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import TokenClassifierOutput

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCHSIZE_TRAIN = 8
BATCHSIZE_VAL = 4
LEARNING_RATE = 5e-5
MAX_LEN = 128
NUM_EPOCH = 20
SEED = 42
NUM_CLASS = 5
MAX_GRAD_NORM = 1
NUM_FILTERS = 10
EMBEDDING_SIZE = 768
window_sizes=(1,2,3,5)


class argu():
    def __init__(self):
        self.dict_path = "./PhoBERT_base_transformers/dict.txt"
        self.config_path = "./PhoBERT_base_transformers/config.json"
        self.max_sequence_length = MAX_LEN
        self.accumulation_steps = 1
        self.epochs = NUM_EPOCH
        self.seed = SEED
        self.bpe_codes = "./PhoBERT_base_transformers/bpe.codes"
args = argu()

config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    return_dict=True,
    num_labels=NUM_CLASS,
    pad_token_id = 1,
    bos_token_id = 0,
    eos_token_id = 2,
    attention_probs_dropout_prob = 0.1,
    classifier_dropout=0.5,
    gradient_checkpointing=False,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-05,
    max_position_embeddings=258,
    model_type="roberta",
    num_attention_heads=12,
    num_hidden_layers=12,
    position_embedding_type="absolute",
    tokenizer_class="PhobertTokenizer",
    transformers_version="4.15.0",
    type_vocab_size=1,
    use_cache=True,
    vocab_size=64001
)

class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, class_weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weight = class_weight
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.class_weight is not None:
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight, dtype=torch.float).to(device))
            else: 
                loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaLSTM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, class_weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weight = class_weight
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=0.5, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            if self.class_weight is not None:
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight, dtype=torch.float).to(device))
            else: 
                loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaCRF(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels[labels==-100] = 0
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)
        tags = tags.to(device)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags

class RobertaLSTMCRF(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=0.5, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            labels[labels==-100] = 0
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)
        tags = tags.to(device)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags


class Roberta_LSTMCRF_CNN(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, alpha=0.01):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.alpha = alpha

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=0.5, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0)) for window_size in window_sizes])
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), 3)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_sent=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits_lstm = self.classifier(lstm_output)

        cls_output = torch.cat((outputs[1][-1][:,0, ...],outputs[1][-2][:,0, ...], outputs[1][-3][:,0, ...], outputs[1][-4][:,0, ...]),-1)

        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(outputs[0].unsqueeze(1)))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)       
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        loss_cls = nn.CrossEntropyLoss()
        if labels_sent is not None:
            loss_cls =  loss_cls(logits, labels_sent.squeeze(1))

        loss = None
        if labels is not None:
            # print(labels)
            labels[labels==-100] = 0
            log_likelihood, tags = self.crf(logits_lstm, labels), self.crf.decode(logits_lstm)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits_lstm)
        tags = torch.Tensor(tags)
        tags = tags.to(device)
        return torch.argmax(logits, dim=1), tags
        # if not return_dict:
        #     output = (tags,) + outputs[2:]
        #     return (((self.alpha * loss)+loss_cls,) + output) if loss is not None else output

        # return (self.alpha * loss)+loss_cls, tags, logits, logits_lstm