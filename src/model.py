import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model 
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.W = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        v, scores = self.attention(q, k, v)

        out = self.concat(v)
        out = self.W(out)

        return out

    
    def split(self, tensor):
        B, seq_len, d_model = tensor.shape
        return tensor.view(B, seq_len, self.n_heads, d_model//self.n_heads).transpose(1, 2)
    
    def concat(self, tensor):
        B, n_heads, seq_len, d_tensor = tensor.shape
        d_model = n_heads * d_tensor

        tensor = tensor.transpose(1, 2)
        tensor = tensor.contiguous().view(B, seq_len, d_model)
        return tensor
    


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None, e=1e-12):
        k_t = k.transpose(2, 3)
        B, heads, seq_len, d_tensor = q.shape

        scores = (q @ k_t)/math.sqrt(d_tensor)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -100000)
        
        scores = self.softmax(scores)
        v = scores @ v
        return v, scores


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-12
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        out = (x - mean) / torch.sqrt(var)
        out = self.gamma * out + self.beta
        return out
    

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, hidden=32):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ff_hidden, n_heads):
        super(EncoderLayer, self).__init__()
        self.ff = FeedForwardLayer(d_model=d_model, hidden=ff_hidden)
        self.mh_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.layer_norm = LayerNorm(d_model=d_model)

    def forward(self, x, src_mask):
        _x = x
        #Attention
        x = self.mh_attn(q=x, k=x, v=x, mask=src_mask)
        #LayerNorm
        x = self.layer_norm(x + _x)
        _x = x
        #FF
        x = self.ff(x)
        x = self.layer_norm(x + _x)
        return x

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, device):
        super().__init__()
        self.emb = TransformerEmbedding(enc_voc_size, d_model, max_len, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                                  ff_hidden=ffn_hidden,
                                                  n_heads=n_head) for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)


        
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        return tok_emb


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, hidden):
        super(DecoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.mh_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ffn = FeedForwardLayer(d_model=d_model, hidden=hidden)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.layer_norm3 = LayerNorm(d_model=d_model)
    

    def forward(self, x_out, x_enc, trg_mask, src_mask):
        _x_out = x_out
        x_out = self.mh_attn(q=x_out, k=x_out, v=x_out, mask=trg_mask)
        x_out = self.layer_norm1(x_out + _x_out)
        _x_out = x_out

        if x_enc is not None: 
            x_out = self.enc_dec_attn(q=x_out, k=x_enc, v=x_enc, mask=src_mask)
            x_out = self.layer_norm2(x_out + _x_out)

        x_out = self.ffn(x_out) 
        x_out = self.layer_norm3(x_out)
        return x_out


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size, d_model=d_model, max_len=max_len, device=device)
        self.layers = nn.ModuleList([DecoderLayer(n_heads=n_head, d_model=d_model, hidden=ffn_hidden) 
                                     for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)
        
    def forward(self, trg, src, trg_mask, src_mask):
        x = self.emb(trg)  # Use embedded input
        for layer in self.layers:
            x = layer(x_out=x, x_enc=src, trg_mask=trg_mask, src_mask=src_mask)
        return self.linear(x)



class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask