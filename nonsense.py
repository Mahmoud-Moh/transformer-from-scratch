import sentencepiece as spm

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
corpus = "This is a lot of stuff that I will write into Nano remember me Linda"
words = corpus.split(" ")
sentences = ["".join(words[ptr:ptr+5]) for ptr in range(len(words)-5-1)]
spm.SentencePieceTrainer.train(
    sentence_iterator=iter(sentences), 
    model_prefix='mymodel',
    pad_id=0, 
    unk_id=1,
    bos_id=2,
    eos_id=3,
    vocab_size=30
)

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))
