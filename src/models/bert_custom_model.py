from .custom_model import CustomModel
from transformers import BertTokenizer, BertModel, BertConfig
import transformers
import torch

class BERTCustomModel(CustomModel):

    def define_model(self, model_to_use):
        self.l1 = transformers.BertModel.from_pretrained(model_to_use)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, len(self.labels))

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
