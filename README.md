# 🔁 Transformer from "Attention Is All You Need" — Reimplemented from Scratch

This project is a reimplementation of the original **Transformer** architecture introduced in the paper [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762).

Special thanks to the repository [**hyunwoongko/transformer**](https://github.com/hyunwoongko/transformer) which served as a guidance source for structuring the codebase and understanding nuances of the implementation.

This repo simplifies the implementation of transformer by eleminiating elements (dropout, bleu accuracy, spacy tokenizer, custom dataLoader, BucketIterator) and embracing native huggingface and pytorch elements (torch DataLoader, transformers.AutoTokenizer) and focusing on the basic concept of transformer.

---

## 📦 Dataset: WMT14 English–German

To download the dataset, use `kagglehub`:

```python
import kagglehub

# Login using your Kaggle API credentials
kagglehub.login()

# Download the EN-DE dataset
path_df_train = kagglehub.dataset_download("mohamedlotfy50/wmt-2014-english-german")
```
Once downloaded, place the contents in the data/ directory.

## 🏋️ Training Setup

- **Model**: Full Transformer (encoder-decoder architecture)
- **Dropout**: Not used
- **Dataset**: WMT14 English–German (only 0.1% used due to resource constraints)
- **Epochs**: 2
- **Framework**: PyTorch
- **Tokenizer**: pretrained t5-base

Despite using only a tiny portion of the dataset, the model successfully learns and reduces the loss over time.
![output](https://github.com/user-attachments/assets/026ab955-372c-4a18-bca3-ae2241bc7cc4)


## 🚀 Quickstart

```bash
# Run training
python train.py
```

