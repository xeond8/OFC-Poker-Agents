import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os

rank_converter = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K",
                  12: "A"}
decode_rank = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}

suit_converter = {0: "d", 1: "h", 2: "s", 3: "c"}

decode_suit = {"d": 0, "h": 1, "s": 2, "c": 3}

class_to_idx = {"empty": 52}
for rank in decode_rank.keys():
    for suit in decode_suit.keys():
        class_to_idx[rank+suit] = decode_suit[suit] * 13 + decode_rank[rank]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=53)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(-1, 64)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CardDataset(Dataset):
    def __init__(self, root_dir="../data/augmentation"):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d for d in os.listdir(root_dir)])
        self.class_to_idx = class_to_idx

        for cls_name in classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        X = torch.unsqueeze(torch.tensor(np.array(Image.open(img_path).convert("L")) / 255, dtype=torch.float32), 0)
        return X, y

if __name__ == '__main__':
    pass