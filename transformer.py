import math
import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class Linear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x):
        x = x @ self.weight
        if "bias" in self:
            x = x + self.bias
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(self, x: mx.array, mask=None):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(3, axis=2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose([0, 2, 1, 3])
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose([0, 2, 1, 3])
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose([0, 2, 1, 3])

        att = (q @ k.transpose([0, 1, 3, 2])) * (1.0 / math.sqrt(k.shape[-1]))

        if mask is not None:
            att = att + mask

        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose([0, 2, 1, 3]).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, affine=config.bias)

    def __call__(self, idx: mx.array, targets: mx.array = None):
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=idx.dtype)  # shape (t)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(idx.shape[1])
        mask = mask.astype(self.wte.weight.dtype)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x, mask)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        else:
            # check to see if this expand_dims is necessary
            logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens=256, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            idx_next = mx.random.categorical(logits * (1 / temperature))
            idx_next = mx.expand_dims(idx_next, axis=0)
            idx = mx.concatenate([idx, idx_next], axis=1)

        return idx
