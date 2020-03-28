from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from sklearn import metrics

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def acc_score(outputs, targets):
    targets = targets.cpu().detach().numpy().tolist()
    outputs = outputs.cpu().detach().numpy().tolist()
    outputs = np.array(outputs) >= 0.5
    return metrics.accuracy_score(targets, outputs)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    _loss = []
    _acc = []
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        mask = d['mask'].to(device)
        targets = d['targets'].to(device)

        optimizer.zero_grad()
        logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(logits, targets)
        _loss.append(loss.cpu().detach().numpy())
        _acc.append(acc_score(torch.sigmoid(logits), targets))
       
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(_loss), np.mean(_acc)

def eval_fn(data_loader, model, device):
    model.eval()
    _loss = []
    _acc = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            mask = d['mask'].to(device)
            targets = d['targets'].to(device)

            logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, targets)
            _loss.append(loss.cpu().detach().numpy())
            _acc.append(acc_score(torch.sigmoid(logits), targets))
        
        return np.mean(_loss), np.mean(_acc)


# def eval_fn(data_loader, model, device):
#     model.eval()
#     fin_targets = []
#     fin_outputs = []
#     with torch.no_grad():
#         for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
#             loss_acc = []
#             ids = d['ids'].to(device)
#             token_type_ids = d['token_type_ids'].to(device)
#             mask = d['mask'].to(device)
#             targets = d['targets'].to(device)

#             outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
#             outputs = torch.sigmoid(outputs)
#             loss = loss_fn(outputs, targets)
#             loss_acc.append(loss.cpu().detach().numpy())

#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
#     return fin_outputs, fin_targets, np.mean(loss_acc)
