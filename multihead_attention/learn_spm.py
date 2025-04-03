import sentencepiece as spm
s = spm.SentencePieceProcessor(model_file='test/test_model.model')
for n in range(5):
	print(s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
