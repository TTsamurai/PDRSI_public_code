import torch.nn as nn
from transformers import AutoModel


class Bert(nn.Module):
    def __init__(self, model_config):
        super(Bert, self).__init__()
        self.bert_type = model_config["bert_type"]
        print("This bert model is {}".format(self.bert_type))

        # Define the mapping from bert_type to pre-trained model name
        model_names = {
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "bertweet": "vinai/bertweet-base",
            "finbert": "ProsusAI/finbert",
        }

        # Load the pre-trained model using the specified bert_type
        if self.bert_type in model_names:
            model_name = model_names[self.bert_type]
            self.model = AutoModel.from_pretrained(model_name)
        else:
            raise ValueError("Unsupported bert_type: {}".format(self.bert_type))

        # Freeze all the parameters of the model
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, inputs):
        outputs = self.model(inputs)
        # TAKE CLS token
        return outputs.pooler_output


# class Bert(nn.Module):
#     def __init__(self, model_config):
#         super(Bert, self).__init__()
#         self.bert_type = model_config["bert_type"]
#         print("This bert model is {}".format(self.bert_type))
#         if self.bert_type == "bert":
#             self.model = BertModel.from_pretrained("bert-base-uncased")
#         elif self.bert_type == "roberta":
#             self.model = RobertaModel.from_pretrained("roberta-base")
#         elif self.bert_type == "bertweet":
#             self.model = AutoModel.from_pretrained("vinai/bertweet-base")
#         elif self.bert_type == "finbert":
#             self.model = AutoModel.from_pretrained("ProsusAI/finbert")
#         for _, param in self.model.named_parameters():
#             param.requires_grad = False

#     def forward(self, inputs):
#         outputs = self.model(inputs)
#         # TAKE CLS token
#         return outputs.pooler_output
