# %%
import config
import transformers
import torch.nn as nn
import torch

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # out1: sequence output  out2: pooling CLS output (1,768)
        out1, out2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out2 = self.bert_drop(out2)
        output = self.out(out2)
        return output



# %% Testing
# tokenizer = config.TOKENIZER
# input = tokenizer.encode_plus('testing the encoder')
# bert_base = BERTBaseUncased()
# ids = torch.LongTensor(input['input_ids']).unsqueeze(0)
# attention = torch.LongTensor(input['attention_mask']).unsqueeze(0)
# token_type_ids = torch.LongTensor(input['token_type_ids']).unsqueeze(0)
# out1, out2 = bert_base(ids, attention, token_type_ids)
# print(out1.shape)
# print(out2.shape)

# %%
