import sentencepiece as spm

class Tokenizer:
    def __init__(self, corpus, vocab_size=37000, words_per_sentences=100, model_prefix="my_model", pad_id=0, unk_id=1,
                 bos_id=2, eos_id=3):
        words = corpus.split(" ")
        sentences = ["".join(words[ptr:ptr+words_per_sentences]) for ptr in range(len(words)-words_per_sentences-1)]
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(sentences), 
            model_prefix=model_prefix,
            pad_id=pad_id, 
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            vocab_size=vocab_size
        )
        encoder = spm.SentencePieceProcessor()
        encoder.load(model_prefix+".model")
        self.encoder = encoder
        
    
    def encode(self, text):
        return self.encoder.encode_as_ids(text)
    