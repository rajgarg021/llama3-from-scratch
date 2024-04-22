import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArguments:
    dim: int = 4096 # input embedding dimension
    n_layers: int = 32 # number of times the transformer block is repeated
    n_heads: int = 32 # number of heads for Queries
    n_kv_heads: Optional[int] = None # number of heads for Keys and Values
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5 # small value added to the denominator in RMSNorm for numerical stability

    max_batch_size: int = 32 # needed for KV cache
    max_seq_length: int = 2048 # needed for KV cache

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def precompute_freqs_complex_exponentials():
    freqs_complex_exponentials = NotImplementedError
    return freqs_complex_exponentials

class RMSNorm(nn.Module):
    """ Root-Mean Squared Normalization """

    def __init__(self):
        raise NotImplementedError

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self):
        raise NotImplementedError

class Transformer(nn.Module):

    def __init__(self, params = ModelArguments) -> None:
        super().__init__()

        assert params.vocab_size != -1, "vocab must be set, cannot be equal to -1"

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        # interspersing communcation and computation by replicating the transformer block sequentially  
        self.layers = nn.Sequential(*[TransformerBlock(params) for _ in range(params.n_layers)])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.lm_head = nn.Linear(params.dim, self.vocab_size)

        self.freqs_complex_exponentials = precompute_freqs_complex_exponentials(self.params.dim // self.params.n_heads, self.params.max_seq_length * 2, device=self.params.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens is a (batch_size (B), sequence_length (T)) tensor of integers
        B, T = tokens.shape
        assert T == 1, "Only one token at a time can be processed"

        # (B, T) --> (B, T, C), C = channel dimension = embedding dimension
        x = self.tok_embeddings(tokens)

        # retrieving the pairs (m, theta) corresponding to the positions [start_pos, start_pos + sequence_length]
        freqs_complex_exponentials = self.freqs_complex_exponentials[start_pos:start_pos + T]

        # sequentially applying all the transformer blocks (decoder layers)
        x = self.layers(x, start_pos, freqs_complex_exponentials) # (B, T , C)
        x = self.norm(x)
        logits = self.lm_head(x).float() # (B, T, vocab_size)

        return logits
    








    


    
