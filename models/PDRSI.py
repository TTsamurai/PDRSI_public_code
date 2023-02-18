import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from utility.dataset import MyDatasetWithBothDiscussion

from model_components.bert import Bert
from model_components.convolution_discussion import (
    ConvDiscussion,
    ConvTechnicalDiscussion,
)


class PDRSI(nn.Module):
    def __init__(self, batch_size, model_config, length_technical):
        super(PDRSI, self).__init__()
        self.max_seq_length = 128
        self.batch_size = batch_size
        self.num_units = 768
        self.num_label = 502
        self.num_heads = 8
        self.hiddent_size = 768
        self.max_history_length = 4
        self.length_technical = length_technical
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
        tweet_label,
        history_label,
        macro_data,
        micro_data,
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


if __name__ == "__main__":
    T_dash = 4
    batch_size = 4
    length_technical = 14
    use_gpu = True
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else "cpu"
    data_train_text_path = "./data/train/text_{}_bert.npy".format(T_dash)
    data_train_label_path = "./data/train/label_{}.npy".format(T_dash)
    data_train_macro_path = "./data/train/macro_{}.npy".format(T_dash)
    data_train_micro_path = "./data/train/micro_{}.npy".format(T_dash)
    data_train_date_path = "./data/train/date_{}.npy".format(T_dash)
    train_text = np.load(data_train_text_path)
    train_label = np.load(data_train_label_path)
    train_macro = np.load(data_train_macro_path)
    train_micro = np.load(data_train_micro_path)
    train_date = np.load(data_train_date_path, allow_pickle=True)
    train_dataset = MyDatasetWithBothDiscussion(
        input_ids=train_text,
        labels=train_label,
        macro_data=train_macro,
        micro_data=train_micro,
        date=train_date,
        length_technical=length_technical,
    )
    right_model_config = {
        "alias": "discussion_text_technical_{}_right_with_timeseires{}_{}_{}".format(
            length_technical, T_dash, "bert", 0
        ),
        "bert_type": "bert",
        "num_epochs": 2,
    }
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # ここを変える
    model = PDRSI(
        batch_size=batch_size,
        model_config=right_model_config,
        length_technical=length_technical,
    )
    print(summary(model=model))

    if use_gpu:
        model.to(device)
    for i, batch in enumerate(train_dataloader):
        input_text = batch[0]
        tweet_label = batch[1].squeeze(dim=1).float()
        history_label = batch[2].float()
        macro_data = batch[3].float()
        micro_data = batch[4].float()
        discussion = batch[5].float()
        technical_discussion = batch[6].float()
        if use_gpu:
            input_text = input_text.to(device)
            tweet_label = tweet_label.to(device)
            history_label = history_label.to(device)
            macro_data = macro_data.to(device)
            micro_data = micro_data.to(device)
            discussion = discussion.to(device)
            technical_discussion = technical_discussion.to(device)
        predict = model(
            input_text,
            tweet_label,
            history_label,
            macro_data,
            micro_data,
            discussion,
            technical_discussion,
        )
        break
    print("success!")
