import torch.nn as nn
from transformers import AutoModel, BertModel, RobertaModel


class Bert(nn.Module):
    def __init__(self, model_config):
        super(Bert, self).__init__()
        self.bert_type = model_config["bert_type"]
        print("This bert model is {}".format(self.bert_type))
        if self.bert_type == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif self.bert_type == "roberta":
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif self.bert_type == "bertweet":
            self.model = AutoModel.from_pretrained("vinai/bertweet-base")
        elif self.bert_type == "finbert":
            self.model = AutoModel.from_pretrained("ProsusAI/finbert")
        # 一旦全てをフリーズさせておく
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, inputs):
        outputs = self.model(inputs)
        # TAKE CLS token
        return outputs.pooler_output
