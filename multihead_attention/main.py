import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from tokenizer import Tokenizer
from dataset import WMT14Dataset
import torch
from torch.utils.data import DataLoader

"""
with open('corpus.txt', 'r') as f:
    corpus = f.read()
print(corpus.split(" "))
tok = Tokenizer(corpus=corpus, vocab_size=85)
print(tok.encode("this is not good at all"))
"""

# Example usage
train_dataset = WMT14Dataset("D:\\gpt\\multihead_attention\\data\\wmt14_translate_de-en_test.csv")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for batch in train_loader:
    x, y = batch
    print("German : ", x )
    print("English: ", y)
    break
