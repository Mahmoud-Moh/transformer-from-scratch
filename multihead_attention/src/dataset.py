import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class WMT14Dataset(Dataset):
    def __init__(self, csv_file, src_lang="de", tgt_lang="en", tokenizer=None):
        self.data = pd.read_csv(csv_file)
        self.src_texts = self.data[src_lang].tolist()
        self.tgt_texts = self.data[tgt_lang].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.src_texts[idx]
        y = self.tgt_texts[idx]
        if self.tokenizer is not None:
            x = self.tokenizer(x)
            y = self.tokenizer(y)
        return x, y
    
