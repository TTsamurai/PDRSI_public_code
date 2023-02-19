import torch
import torch.nn as nn

from model_components.bert import Bert
from model_components.convolution_discussion import (
    ConvDiscussion,
    ConvTechnicalDiscussion,
)


class PDRSI(nn.Module):
    def __init__(self, model_config):
        super(PDRSI, self).__init__()
        self.max_seq_length = 128
        self.batch_size = model_config["batch_size"]
        self.num_units = 768
        self.num_label = 502
        self.num_heads = 8
        self.hiddent_size = 768
        self.max_history_length = 4
        self.length_technical = model_config["length_technical"]
        self.model = Bert(model_config=model_config)
        self.linear_prediction = nn.Sequential(
            nn.Linear(self.hiddent_size, self.num_label), nn.Dropout(p=0.1)
        )
        self.multihead_torch = nn.MultiheadAttention(
            embed_dim=self.num_units,
            num_heads=self.num_heads,
            vdim=self.num_label,
            batch_first=True,
        )
        self.conv_discussion = ConvDiscussion()  # input
        self.rnn_model = nn.LSTM(
            self.hiddent_size, self.hiddent_size, 1, batch_first=True
        )
        self.conv_technical_discussion = ConvTechnicalDiscussion(
            n_length=self.length_technical
        )

    def forward(
        self,
        input_ids,
        history_label,
        discussion,
        technical_disccusion,
    ):
        discussion_embed = self.conv_discussion(discussion)
        technical_discussion_embed = self.conv_technical_discussion(
            technical_disccusion
        )
        # Baselineモデルではhistory
        history_label = history_label[:, 1:, :]
        input_ids_compress = input_ids.view(-1, self.max_seq_length)
        bert_output = self.model(input_ids_compress)
        if input_ids.size()[0] == self.batch_size:
            reshape_bert_output = bert_output.view(
                self.batch_size, (1 + self.max_history_length), -1
            )
        else:

            reshape_bert_output = bert_output.view(
                input_ids.size()[0], (1 + self.max_history_length), -1
            )
        rnn_input = reshape_bert_output.flip(dims=[1])
        output, (time_series_emb, _) = self.rnn_model(rnn_input)
        time_series_emb = time_series_emb.view(input_ids.size()[0], -1)
        tweet_emb = reshape_bert_output[:, :1, :]
        history_emb = reshape_bert_output[:, 1:, :]

        attention_output, attention = self.multihead_torch.forward(
            query=tweet_emb, key=history_emb, value=history_label
        )

        output = torch.squeeze(attention_output, dim=1)
        output = (
            discussion_embed + technical_discussion_embed + output + time_series_emb
        )
        predict_score = self.linear_prediction(output)
        return predict_score
