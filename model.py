import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArguments:
    n_embeddings: int = 4096 # input embedding dimension
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

def precompute_freqs_complex(head_size: int, seq_len: int, device: str, theta: float = 10000.0):
    """ Precomputing the frequency tensor with complex exponentials for the given sequence length and dimensions """

    assert head_size % 2 == 0, "embedding dimension must be even"
    # first computing the theta parameter according to the formula in paper
    # theta_i = 10000 ^ (-2 * (i-1)/d) for i = [1, 2, 3, .... , d/2]
    theta = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size)).to(device) # (head_size/2)
    # now computing the m (positions) paƒrameter
    m = torch.arange(seq_len, device=device) # (seq_len)
    # using outer product to multiply each theta with each m (positions)
    freqs = torch.outer(m, theta).float() # (seq_len) * (head_size/2) --> (seq_len, head_size/2)
    # now converting the above frequencies into complex numbers using the polar form c = R * exp(i * m * theta), where R = 1
    freqs_complex =  torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_size/2)

    return freqs_complex

def apply_rotary_position_embeddings(x: torch.tensor, freqs_complex: torch.tensor, device: str):
    """  Applying rotary position embeddings to input tensors using the given frequency tensor """

    # converting two consecutive numbers into a single complex number
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # (B, seq_len, H, head_size/2)
    # reshaping freqs_complex tensor to match the shape of x_complex tensor in order to perform element-wise operations
    # adding the batch and head dimensions
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (seq_len, head_size/2) --> (1, seq_len, 1, head_size/2)
    x_rotated = x_complex * freqs_complex # (B, seq_len, H, head_size/2) * (1, seq_len, H, head_size/2) --> (B, seq_len, H, head_size/2)
    # converting the above rotated tensor into a new tensor where the first element is the real part and the second element is the imaginary part of the complex number
    x_out = torch.view_as_real(x_rotated) # (B, seq_len, H, head_size/2) --> (B, seq_len, H, head_size/2, 2)
    # flattening the above tensor
    x_out = x_out.reshape(*x.shape) # (B, seq_len, H, head_size/2, 2) --> (B, seq_len, H, head_size)
    
    return x_out.type_as(x).to(device)

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
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.n_embeddings)

        # interspersing communcation and computation by replicating the transformer block sequentially  
        self.layers = nn.Sequential(*[TransformerBlock(params) for _ in range(params.n_layers)])

        self.norm = RMSNorm(params.n_embeddings, eps=params.norm_eps)
        self.lm_head = nn.Linear(params.n_embeddings, self.vocab_size)

        # note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096
        # adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning
        self.freqs_complex = precompute_freqs_complex(self.params.n_embeddings // self.params.n_heads, self.params.max_seq_length * 2, device=self.params.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens is a (batch_size (B), sequence_length (seq_len)) tensor of integers
        B, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"

        x = self.tok_embeddings(tokens) # (B, seq_len) --> (B, seq_len, n_embeddings)

        # retrieving the pairs (m, theta) corresponding to the positions [start_pos, start_pos + sequence_length]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # sequentially applying all the transformer blocks (decoder layers)
        x = self.layers(x, start_pos, freqs_complex) # (B, seq_len , n_embeddings)
        x = self.norm(x) # (B, seq_len , n_embeddings)
        logits = self.lm_head(x).float() # (B, seq_len, vocab_size)

        return logits
    








    


    
