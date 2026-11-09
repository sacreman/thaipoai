"""A bilingual encoder-decoder Transformer in core MLX.

The source and target embeddings and the output projection share one matrix, which is trained like
every other weight.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from train.tokenizer import BOS, EOS, PAD


@dataclass(frozen=True)
class Config:
    vocab: int
    dim: int
    heads: int
    feed_forward: int
    layers: int
    dropout: float = 0.1
    max_length: int = 128


CONFIGS = {
    "full": Config(vocab=32000, dim=1024, heads=16, feed_forward=4096, layers=9),
    "baseline": Config(vocab=32000, dim=512, heads=8, feed_forward=2048, layers=8),
}


def positions(length, dim):
    """Sinusoidal positions. Kept out of the parameters, so they are never trained or saved."""
    position = mx.arange(length, dtype=mx.float32)[:, None]
    frequency = mx.exp(mx.arange(0, dim, 2, dtype=mx.float32) * (-math.log(10000.0) / dim))
    table = mx.zeros((length, dim))
    table[:, 0::2] = mx.sin(position * frequency)
    table[:, 1::2] = mx.cos(position * frequency)[:, : dim // 2]
    return table


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.scale = heads, math.sqrt(heads / dim)
        self.query, self.key, self.value, self.out = [nn.Linear(dim, dim, bias=False) for _ in range(4)]

    def split(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, -1).transpose(0, 2, 1, 3)

    def __call__(self, query, memory, mask):
        result = mx.fast.scaled_dot_product_attention(
            self.split(self.query(query)), self.split(self.key(memory)), self.split(self.value(memory)), scale=self.scale, mask=mask
        )
        return self.out(result.transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], -1))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up, self.down = nn.Linear(config.dim, config.feed_forward), nn.Linear(config.feed_forward, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        return self.down(self.dropout(nn.gelu(self.up(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention, self.feed_forward = Attention(config.dim, config.heads), FeedForward(config)
        self.norm1, self.norm2 = nn.LayerNorm(config.dim), nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x, mask):
        normed = self.norm1(x)
        x = x + self.dropout(self.attention(normed, normed, mask))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention, self.cross_attention = Attention(config.dim, config.heads), Attention(config.dim, config.heads)
        self.feed_forward = FeedForward(config)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(config.dim), nn.LayerNorm(config.dim), nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, y, memory, target_mask, memory_mask):
        normed = self.norm1(y)
        y = y + self.dropout(self.self_attention(normed, normed, target_mask))
        y = y + self.dropout(self.cross_attention(self.norm2(y), memory, memory_mask))
        return y + self.dropout(self.feed_forward(self.norm3(y)))


class Translator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.dim)
        # The default scale would make the shared output projection wildly overconfident at step one.
        self.embedding.weight = mx.random.normal((config.vocab, config.dim), scale=config.dim**-0.5)
        self.encoder = [EncoderLayer(config) for _ in range(config.layers)]
        self.decoder = [DecoderLayer(config) for _ in range(config.layers)]
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(config.dim), nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self._positions = positions(config.max_length, config.dim)

    def embed(self, ids):
        scaled = self.embedding(ids) * math.sqrt(self.config.dim)
        return self.dropout(scaled + self._positions[: ids.shape[1]])

    def encode(self, source):
        mask = (source != PAD)[:, None, None, :]
        x = self.embed(source)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.encoder_norm(x), mask

    def decode(self, target, memory, memory_mask):
        length = target.shape[1]
        mask = mx.tril(mx.ones((length, length), dtype=mx.bool_))[None, None] & (target != PAD)[:, None, None, :]
        y = self.embed(target)
        for layer in self.decoder:
            y = layer(y, memory, mask, memory_mask)
        return self.embedding.as_linear(self.decoder_norm(y))

    def __call__(self, source, target):
        memory, memory_mask = self.encode(source)
        return self.decode(target, memory, memory_mask)

    def greedy(self, source, limit=None):
        """Translate one encoded source. Recomputes the prefix each step, so it is for checks, not latency."""
        self.eval()
        memory, mask = self.encode(mx.array([source]))
        output = [BOS]
        for _ in range((limit or self.config.max_length) - 1):
            logits = self.decode(mx.array([output]), memory, mask)
            token = logits[0, -1].argmax().item()
            if token == EOS:
                break
            output.append(token)
        self.train()
        return output[1:]


def parameter_count(model):
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
