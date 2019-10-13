import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class BaseModel(nn.Module):
    def __init__(self, args, use_gpu):
        super(BaseModel, self).__init__()
        self.num_labels = 2
        self.use_gpu = use_gpu

        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(768, self.num_labels)
        self.init_weight()

    def forward(self, ids, type, mask):
        _, pooled_output = self.bert_model(ids, type, mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def init_weight(self):
        self.classifier.bias.data.zero_()




