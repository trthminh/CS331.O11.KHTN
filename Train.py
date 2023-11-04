import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, RandomRotation
import numpy as np
from torch.utils.data import DataLoader
from Dataset import PixWiseDataset
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
from Trainer import Trainer
import pandas as pd

import os

#------------------------------------
train_csv_path = './data/train_skip_frames/train_skip_frames.csv'
train_ratio = 0.8
batch_size = 4
name_model = './DeePixBiS.pth'
#------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# name_model = './DeePixBiS_train_median_frame.pth'
model = DeePixBiS()
# Nếu train lần 2 thì bật lệnh này, lần đầu thì thôi
# model.load_state_dict(torch.load(name_model))

model.eval()

loss_fn = PixWiseBCELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.0001)


train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]), 
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

df = pd.read_csv(train_csv_path)
msk = np.random.rand(len(df)) < train_ratio
train_df = df[msk]
val_df = df[~msk]

train_dataset = PixWiseDataset(train_df, transform=train_tfms)
train_ds = train_dataset.dataset()

val_dataset = PixWiseDataset(val_df, transform=test_tfms)
val_ds = val_dataset.dataset()

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)


trainer = Trainer(train_dl, val_dl, model, 5, opt, loss_fn, device=device)

print('Training Beginning\n')
trainer.fit()
print('\nTraining Complete')
torch.save(model.state_dict(), name_model)