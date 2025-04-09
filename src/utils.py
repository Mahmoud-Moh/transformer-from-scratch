import os 
from dataset import WMT14Dataset
from torch.utils.data import DataLoader

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_data(data_path, batch_size, tokenizer=None, shuffle=True, data_percentage=0.1):
    train_pth = os.path.join(data_path, "wmt14_translate_de-en_train.csv") 
    test_pth = os.path.join(data_path, "wmt14_translate_de-en_test.csv") 
    val_pth = os.path.join(data_path, "wmt14_translate_de-en_validation.csv")  

    #load data
    train_data = WMT14Dataset(csv_file=train_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer, data_percentage=data_percentage)
    test_data = WMT14Dataset(csv_file=test_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer, data_percentage=data_percentage)
    val_data = WMT14Dataset(csv_file=val_pth, src_lang="de", tgt_lang="en", tokenizer=tokenizer, data_percentage=data_percentage)
    #data loaders
    train_dl = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle)
    val_dl = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=shuffle)

    return train_dl, test_dl, val_dl
