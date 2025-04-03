# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import kagglehub
import os
kagglehub.login()

# Set the path to the file you'd like to load
#train_path = "wmt14_translate_de-en_train.csv"
#test_path = "wmt14_translate_de-en_test.csv"
#val_path = "wmt14_translate_de-en_validation.csv"

# Load the latest version
path_df_train = kagglehub.dataset_download(
  "mohamedlotfy50/wmt-2014-english-german")

print("Path to train data:", path_df_train)


root = "C:\\Users\\mahmoud\\.cache\\kagglehub\\datasets\\mohamedlotfy50\\wmt-2014-english-german\\versions\\1"
train_name = "wmt14_translate_de-en_train.csv"
test_name = "wmt14_translate_de-en_test.csv"
val_name = "wmt14_translate_de-en_validation.csv"

train_pth = os.path.join(root, train_name)
test_pth = os.path.join(root, test_name)
val_name = os.path.join(root, val_name)

#train_path = "datasets/train.csv"
#test_path = "datasets/test.csv"
#val_path = "datasets/val.csv"

#df_train.to_csv(train_path)
#df_test.to_csv(test_path)
#df_val.to_csv(val_path)
