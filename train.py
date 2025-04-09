"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import sys
from torch import nn, optim
from torch.optim import Adam
from transformers import AutoTokenizer
import os
import yaml
import torch
import math
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dataset import * 
from utils import epoch_time, load_data
from model import *
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
DATA_PATH = os.path.join(os.path.dirname(__file__),  "data")

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
BATCH_SIZE = int(config["training"]["batch_size"])
EPOCHS = int(config["training"]["epochs"])
MAX_LENGTH = int(config["training"]["max_length"])
SHUFFLE = bool(config["training"]["shuffle"])
D_MODEL = int(config["training"]["d_model"])
N_HEADS = int(config["training"]["n_heads"])
FFN_HIDDEN = int(config["training"]["ffn_hidden"])
N_LAYERS = int(config["training"]["n_layers"])
INIT_LR = float(config["training"]["init_lr"])
WEIGHT_DECAY = float(config["training"]["weight_decay"])
ADAM_EPS = float(config["training"]["adam_eps"])
LR_SCHEDULE_FACTOR = float(config["training"]["lr_scheduler"]["factor"])
LR_SCHEDULE_PATIENCE = int(config["training"]["lr_scheduler"]["patience"])
DATA_PERCENTAGE = float(config["training"]["data_percentage"])


DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.add_special_tokens({'bos_token': '<bos>'})
bos_token_id = tokenizer.convert_tokens_to_ids('<bos>')


model = Transformer(src_pad_idx=tokenizer.pad_token_id,
                    trg_pad_idx=tokenizer.pad_token_id,
                    trg_sos_idx=bos_token_id,
                    d_model=D_MODEL,
                    enc_voc_size=len(tokenizer),
                    dec_voc_size=len(tokenizer),
                    max_len=MAX_LENGTH,
                    ffn_hidden=FFN_HIDDEN,
                    n_head=N_HEADS,
                    n_layers=N_LAYERS,
                    device=DEVICE).to(DEVICE)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=INIT_LR,
                 weight_decay=WEIGHT_DECAY,
                 eps=ADAM_EPS)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=LR_SCHEDULE_FACTOR,
                                                 patience=LR_SCHEDULE_PATIENCE)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch["input_ids"].to(DEVICE)
        trg = batch["label"].to(DEVICE)
        decoded_src = tokenizer.decode(src[0], skip_special_tokens=True)
        decoded_trg = tokenizer.decode(trg[0], skip_special_tokens=True)
        #print("src----> ", decoded_src)
        #print("trg-----> ", decoded_trg)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        #print("type op:: ", type(output_reshape))
        #print("type op[0]::", type(output_reshape[0]))
        #print("type in type :: ", type(output_reshape[0][0]))
        #print(output_reshape[0])
        #decoded_op = tokenizer.decode(output_reshape[0], skip_special_tokens=True)
        #print("decoded_op------> ", decoded_op)
        #time.sleep(5)

        loss = criterion(output_reshape, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch["input_ids"].to(DEVICE)
            trg = batch["label"].to(DEVICE)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()


    return epoch_loss / len(iterator)


def run(total_epoch, train_iter, valid_iter, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')

if __name__ == '__main__':
    data_path = ""
    if os.name == 'nt':
        data_path = "D:\\gpt\\data"
    elif os.name == 'posix':
        data_path = "/mnt/d/gpt/data"
    train_dl, test_dl, val_dl = load_data(data_path=data_path, tokenizer=tokenizer, batch_size=BATCH_SIZE, data_percentage=DATA_PERCENTAGE)
    run(total_epoch=EPOCHS, train_iter=train_dl, valid_iter=val_dl, best_loss=math.inf)
