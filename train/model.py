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
    max_length: int = 256  # the trainer's longest bucket


CONFIGS = {
    "full": Config(vocab=32000, dim=1024, heads=16, feed_forward=4096, layers=9),
    "baseline": Config(vocab=32000, dim=512, heads=8, feed_forward=2048, layers=8),
    "probe": Config(vocab=32000, dim=256, heads=4, feed_forward=1024, layers=6),  # about 19M, for quick comparisons
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

    def __call__(self, query, memory, mask, cache=None, grow=True):
        """cache holds earlier keys and values. Self-attention grows it, and cross-attention reuses it."""
        if cache is not None and not grow:
            keys, values = cache
        else:
            keys, values = self.split(self.key(memory)), self.split(self.value(memory))
            if cache is not None:
                keys, values = mx.concatenate([cache[0], keys], axis=2), mx.concatenate([cache[1], values], axis=2)
        result = mx.fast.scaled_dot_product_attention(self.split(self.query(query)), keys, values, scale=self.scale, mask=mask)
        return self.out(result.transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], -1)), (keys, values)


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
        x = x + self.dropout(self.attention(normed, normed, mask)[0])
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention, self.cross_attention = Attention(config.dim, config.heads), Attention(config.dim, config.heads)
        self.feed_forward = FeedForward(config)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(config.dim), nn.LayerNorm(config.dim), nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, y, memory, target_mask, memory_mask, cache=None):
        past_self, past_cross = cache or (None, None)
        normed = self.norm1(y)
        attended, self_cache = self.self_attention(normed, normed, target_mask, past_self)
        y = y + self.dropout(attended)
        attended, cross_cache = self.cross_attention(self.norm2(y), memory, memory_mask, past_cross, grow=False)
        y = y + self.dropout(attended)
        return y + self.dropout(self.feed_forward(self.norm3(y))), (self_cache, cross_cache)


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

    def embed(self, ids, start=0):
        scaled = self.embedding(ids) * math.sqrt(self.config.dim)
        return self.dropout(scaled + self._positions[start : start + ids.shape[1]])

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
            y, _ = layer(y, memory, mask, memory_mask)
        return self.embedding.as_linear(self.decoder_norm(y))

    def next_token(self, last, position, memory, memory_mask, caches):
        """One decoding step: only the newest token is processed, so a sentence costs linear time."""
        y = self.embed(last, position)
        updated = []
        for layer, cache in zip(self.decoder, caches or [None] * len(self.decoder)):
            y, cache = layer(y, memory, None, memory_mask, cache)
            updated.append(cache)
        return self.embedding.as_linear(self.decoder_norm(y))[:, -1].argmax(-1).astype(mx.int32), updated

    def __call__(self, source, target):
        memory, memory_mask = self.encode(source)
        return self.decode(target, memory, memory_mask)

    def translate(self, sources, limit=None, batch_size=64):
        """Greedy translation of encoded sources, in batches sorted by length."""
        self.eval()
        results = [None] * len(sources)
        order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
        for start in range(0, len(order), batch_size):
            group = order[start : start + batch_size]
            longest = len(sources[group[-1]])
            source = mx.array([sources[i] + [PAD] * (longest - len(sources[i])) for i in group], mx.int32)
            memory, mask = self.encode(source)
            mx.eval(memory)
            token, caches, finished, steps = mx.full((len(group),), BOS, mx.int32), None, mx.zeros((len(group),), mx.bool_), []
            for position in range(min(limit or longest * 2 + 16, self.config.max_length) - 1):
                token, caches = self.next_token(token[:, None], position, memory, mask, caches)
                token = mx.where(finished, mx.array(PAD, mx.int32), token)
                finished = finished | (token == EOS)
                mx.eval(token, finished, caches)
                steps.append(token)
                if finished.all().item():
                    break
            output = mx.stack(steps, axis=1).tolist()
            for index, ids in zip(group, output):
                results[index] = [i for i in (ids[: ids.index(EOS)] if EOS in ids else ids) if i not in (PAD, BOS)]
        self.train()
        return results


def parameter_count(model):
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
