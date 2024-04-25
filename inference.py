import torch
import time
import json
from pathlib import Path

from model import ModelArgs, Transformer
from tokenizer import Tokenizer

class Llama:
    """ Llama class """

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        """ Building a Llama instance by initializing and loading a model checkpoint """

        # loading the model
        if load_model:
            # loading the checkpoint
            start_time = time.time()
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            checkpoint_path = checkpoints[0]
            print(f"Loading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - start_time):.2f}s")
        
        start_time = time.time()
        # building the parameters (model arguments)
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
            )

        # loading the tokenizer
        tokenizer = Tokenizer(tokenizer_path)
        # using the tokenizer to populate the vocabulary size of our model
        model_args.vocab_size = tokenizer.n_words

        # setting the default torch tensor type 
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model = Transformer(model_args).to(device)
        if load_model:
            # since we are computing RoPE during inference, we don't need the frequencies provided by META
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - start_time):.2f}s")

        return Llama(model, model_args, tokenizer)
    
    def _sample_top_p(self, probs, p):
        """ Performing top-p (nucleus) sampling on a given probability distribution """

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1) # (B, vocab_size)
        # substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
        mask = probs_sum - probs_sort > p # (B, vocab_size)
        # zeroing out all the probabilities of tokens that are not selected by the top p
        probs_sort[mask] = 0.0 
        # redistributing the probabilities so that they sum up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # sampling a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # gathering the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token



if __name__ == "__main__":
    torch.manual_seed(42)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available and allow_cuda else "cpu"

    prompts = [
        ""
    ]

    model = Llama.build(
        checkpoints_dir="Meta-Llama-3-8B/",
        tokenizer_path="Meta-Llama-3-8B/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

print("ALL GOOD! :)")
