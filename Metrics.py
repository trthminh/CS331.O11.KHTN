import torch
import numpy as np
from tqdm import tqdm

def predict(mask, label, threshold=0.5, score_type="combined"):
    with torch.no_grad():
        if score_type == "pixel":
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == "binary":
            score = label
        else:
            score = (torch.mean(mask, axis=(1, 2, 3)) + label) / 2

        preds = (score > threshold).type(torch.FloatTensor)

        return preds, score

def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()

def test_accuracy(model, test_dl):
    acc = 0
    total = len(test_dl)
    for i, (img, mask, label) in enumerate(test_dl):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img, mask, label = img.to(device), mask.to(device), label.to(device)
        net_mask, net_label = model(img)

        # Calculate predictions accuracy
        preds, score = predict(net_mask, net_label)
        targets, _ = predict(mask, label)
        current_acc = calc_acc(preds, targets)
        acc += current_acc
    return acc / total


def test_loss(model, test_dl, loss_fn):
    loss = 0
    total = len(test_dl)
    for i, (img, mask, label) in enumerate(test_dl):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img, mask, label = img.to(device), mask.to(device), label.to(device)
        net_mask, net_label = model(img)

        # calculate loss
        losses = loss_fn(net_mask, net_label, mask, label)
        loss += torch.mean(losses).item()
    return loss / total