import torch
from torch import nn
import torch.nn.functional as F



def clip_loss(source, target):
    # Getting Image and Text Features
    
    temperature =1

    # Calculating the Loss
    logits = torch.matmul(source, target.transpose(-1, -2)) 
    source_similarity = torch.matmul(source, source.transpose(-1, -2))
    target_similarity = torch.matmul(target, target.transpose(-1, -2))
    targets = F.softmax(
        (source_similarity + target_similarity) / 2 , dim=-1
    )
    source_loss = cross_entropy(logits, targets, reduction='none')
    target_loss = cross_entropy(logits.transpose(-1, -2), targets.transpose(-1, -2), reduction='none')
    loss =  (source_loss + target_loss) / 2.0 # shape: (batch_size)
    return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
