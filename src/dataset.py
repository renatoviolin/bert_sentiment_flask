# %%
import torch
import config

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, index):
        review = str(self.review[index])
        review = ' '.join(review.split())
        inputs = self.tokenizer.encode_plus(review, max_length=self.max_len, pad_to_max_length=True)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # padding_len = self.max_len - len(ids)
        # ids = ids + ([0] * padding_len)
        # mask = mask + ([0] * padding_len)
        # token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'token_type_ids': torch.LongTensor(token_type_ids),
            'targets': torch.tensor(self.target[index], dtype=torch.float)
        }


# %%
# review = ['the movie was good', 'movie awaful']
# target = [1.0, 0.0]
# ds = BERTDataset(review, target)
# print(ds[1]['targets'])

# %%
