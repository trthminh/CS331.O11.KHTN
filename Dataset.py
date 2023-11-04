import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from random import shuffle
import pandas as pd
import os


class PixWiseDataset():
    def __init__(self, df, map_size=14,
                 smoothing=True, transform=None):
        self.data = df
        self.transform = transform
        self.map_size = map_size
        self.label_weight = 0.99 if smoothing else 1.0


    def dataset(self):
        images = []
        labels = []
        masks = []
        for idx, row in self.data.iterrows():
            img_name = row["name"]
            img = Image.open(img_name)

            label = row["label"]
            if label == 0:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
            else:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

            if self.transform:
                img = self.transform(img)

            images.append(img)
            labels.append(label)
            masks.append(mask)
        labels = np.array(labels, dtype=np.float32)

        dataset = [[images[i], masks[i], labels[i]] for i in range(len(images))]
        return dataset