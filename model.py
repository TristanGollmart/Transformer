# coding a transformer architecture following the original paper, using their naming conventions

import torch
import torch.nn as nn
import math

class InputEmdbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return  self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create matrix for encoding of shape [seq_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        # Create vector of shape (seq_len) to encode position of word in sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqeeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe.unsqueeze(0) # for batches

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float=10**-6):
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.Linear(d_model, d_ff)
        self.ln2 = nn.Linear(d_ff, d_model) # return shape same as input shape

    def forward(self, x):
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        self.d_model = d_model
        self.h = h # number of attention heads

        assert d_model % h == 0, "model dimension must be divisible by number of heads"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # query
        self.w_k = nn.Linear(d_model, d_model) # keys
        self.w_v = nn.Linear(d_model, d_model) # values
        self.w_o = nn.Linear(d_model, d_model) # final layer after each head concatenated
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # query, key, value: shape [B, H, T, d_k]
        # returns: single head attentions and attention scores for model interpretation
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask==0, -torch.inf)
        attention_scores = attention_scores.softmax(dim=-1) # B, H, T, T
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (B, H, T, D_K)
        return attention_scores @ value, attention_scores


    def forward(self, q, k, v, mask):
        keys = self.w_k(k)
        queries = self.w_q(q)
        values = self.w_v(v)

        # split into sub-embeddings for the h single heads each of width d_k
        # transpose to have small split matrices in last 2 dimensions for applying multiplication Q*KT
        queries = queries.view(queries.shape[0], queries.shape[1], self.h, self.d_k).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.h, self.d_k).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(queries, keys, values, mask, self.dropout)

        # (B, H, T, D_K) --> # (B, T, H, D_K) --> (B, T, D_Model)
        x = x.transpose(1, 2)
        x = torch.concatenate(x, dim=2)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)  # to do inplace
        x = self.w_o(x)
        return x


class ResidualConnection(nn.Module)
    def __init__(self, dropout: float):
        super.__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# blocks of the encoder repeated N times before sent to decoder
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super.__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)


    def forward(self, x, src_mask):
        # src_mask: mask for input to encoder to mask padding of inputs
        x = self.residual_connections1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections2(x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init(self, layers: nn.ModuleList):
        super.__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ---------------------------------------
# ------------ Decoder ------------------


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # self.dropout = nn.Dropout(dropout)
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)
        self.residual_connection3 = ResidualConnection(dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # src_mask: mask from encoder
        # trg_mask: mask from decoder for target language in machine translation
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connection2(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection3(x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (B, T, D_Model) --> (B, T, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embded: InputEmdbedding, trg_embed: InputEmdbedding,
                 src_pos: PositionalEncoding, trg_pos: PositionalEncoding, proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embded
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.proj_layer(x)