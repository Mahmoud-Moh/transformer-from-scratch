import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from model import MultiHeadAttention
from dataset import WMT14Dataset
from torch.utils.data import DataLoader
import yaml 
import sentencepiece as spm
import torch
from tokenizer import Tokenizer
from transformers import AutoTokenizer
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
EPOCHS = config["training"]["epochs"]
MAX_LENGTH = config["training"]["max_length"]
SHUFFLE = config["training"]["shuffle"]
D_MODEL = config["training"]["d_model"]
N_HEADS = config["training"]["n_heads"]

#=========================
#Load data 
#=========================
def load_data(data_path, batch_size, tokenizer=None, shuffle=True):
    train_pth = os.path.join(data_path, "wmt14_translate_de-en_train.csv") 
    test_pth = os.path.join(data_path, "wmt14_translate_de-en_test.csv") 
    val_pth = os.path.join(data_path, "wmt14_translate_de-en_validation.csv")  

    #load data
    train_data = WMT14Dataset(csv_file=train_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer)
    test_data = WMT14Dataset(csv_file=test_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer)
    val_data = WMT14Dataset(csv_file=val_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer)

    #data loaders
    train_dl = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle)
    val_dl = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=shuffle)

    return train_dl, test_dl, val_dl

#=========================
#Load model
#=========================
def load_model(d_model, n_heads):
    model = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    return model

#=========================
#training functions 
#=========================
def loss_batch(loss_fn, xb, yb, yb_h, opt=None):
    loss = loss_fn(yb_h, yb)
    #metric_b = metics_batch(yb, yb_h)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), None

def loss_epoch(model, loss_fn, dataset_dl, device='cpu', opt=None):
    loss = 0.0
    metric = 0.0
    len_data = len(dataset_dl.dataset)
    for batch in dataset_dl:
        src = batch["input_ids"]
        trg = batch["label"]
        output = model(src, trg)
        #Note that src and trg are of length B * max_len * vocab_size 
        #For example 4 * 256 * 37000


        xb = xb.to(device)
        yb = yb.to(device)
        yb_h = model(xb)

        loss_b, metric_b = loss_batch(loss_fn, xb, yb, yb_h, opt)
        loss += loss_b
        if metric_b is not None:
            metric_b += metric_b
    loss /= len_data
    metric /= len_data
    return loss, metric

def train_val(epochs, model, loss_fn, opt, train_dl, val_dl, device, sanity_check=False):
    for epoch in range(epochs):
        loss_train, metric_train = loss_epoch(model, loss_fn, train_dl, device, opt)
        model.eval()
        with torch.no_grad():
            loss_val, metric_val = loss_epoch(model, loss_fn, train_dl, device, opt)
        print("Epoch %d: Train loss %.4f, Train metric %.4f Val Loss %.4f, Val Metric %.4f"
              % (epoch, loss_train, metric_train, loss_val, metric_val))


if __name__ == "__main__":
    data_path = ""
    if os.name == 'nt':
        data_path = "D:\\gpt\\multihead_attention\\data"
    elif os.name == 'posix':
        data_path = "/mnt/d/gpt/multihead_attention/data"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dl, test_dl, val_dl = load_data(data_path=data_path, tokenizer=tokenizer, batch_size=4)


