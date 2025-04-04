import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class WMT14Dataset(Dataset):
    def __init__(self, csv_file, max_length=512, src_lang="de", tgt_lang="en", tokenizer=None):
        self.data = pd.read_csv(csv_file, lineterminator="\n")
        self.src_texts = self.data[src_lang].tolist()
        self.tgt_texts = self.data[tgt_lang].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.src_texts[idx]
        y = self.tgt_texts[idx]
        if self.tokenizer is not None:
            encoding_x = self.tokenizer(
                x, 
                truncation=True, 
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoding_y = self.tokenizer(
                y, 
                truncation=True, 
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        return {
            "input_ids": encoding_x["input_ids"].squeeze(0),
            "attention_mask": encoding_x["attention_mask"].squeeze(0),
            "label": encoding_y["input_ids"].squeeze(0)
        }
    
