import config
import torch
import pandas as pd
import dataset
from model import BERTBaseUncased
from sklearn.model_selection import train_test_split
import engine
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

def run():
    df = pd.read_csv(config.TRAIN_FILE).fillna('none')
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
    df_train, df_valid = train_test_split(df,
                                          test_size=0.1,
                                          random_state=42,
                                          stratify=df.sentiment.values)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(review=df_train.review.values,
                                        target=df_train.sentiment.values)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.BERTDataset(review=df_valid.review.values,
                                        target=df_valid.sentiment.values)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_acc = 0
    print(f'cuda: {torch.cuda.is_available()}')
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss, valid_acc = engine.eval_fn(valid_data_loader, model, device)
        
        print(f'Train loss: {train_loss:.4f}\t Train acc. {train_acc:.2f}')
        print(f'Valid loss: {valid_loss:.4f}\t Valid acc. {valid_acc:.2f}')
        if train_acc > best_acc:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_acc = train_acc


if __name__ == '__main__':
    run()
