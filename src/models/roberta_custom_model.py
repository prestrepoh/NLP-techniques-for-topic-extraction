from .custom_model import CustomModel
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import transformers
import torch

class RoBERTaCustomModel(CustomModel):

    def define_model(self, model_to_use):
        self.l1 = transformers.RobertaModel.from_pretrained(model_to_use)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, len(self.labels))

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state = self.l1(input_ids = ids, token_type_ids = token_type_ids, attention_mask= mask).last_hidden_state
        output_1 = self.pool_hidden_state(last_hidden_state)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
    def pool_hidden_state(self,last_hidden_state):
      return torch.mean(last_hidden_state, 1)