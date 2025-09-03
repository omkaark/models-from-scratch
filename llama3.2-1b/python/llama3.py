from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json


tokenizer_path = "../model/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)


model = torch.load("../model/consolidated.00.pth", map_location=torch.device('cpu'))

with open("../model/params.json", "r") as f:
    config = json.load(f)

embd = torch.nn.Embedding(tokenizer.n_vocab, config['dim'])
embd.load_state_dict({'weight': model['tok_embeddings.weight']})

prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
tokens = torch.tensor(tokens)

SEQ_LEN = 1024
H = 64

freqs = 1.0 / (config['rope_theta'] ** (torch.arange(0, H, 2).float() / H))
freqs_idx = torch.outer(torch.arange(SEQ_LEN), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_idx), freqs_idx)

def apply_rope(x):
    """Apply rotary positional encoding to queries or keys"""
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (config['rope_theta'] ** (torch.arange(0, H, 2).float() / H))
    freqs_idx = torch.outer(torch.arange(SEQ_LEN), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_idx), freqs_idx)
    x_split = x.float().view(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_split)
    x_rotated = x_complex * freqs_cis[:seq_len]
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

while tokens[-1] != 128001:
    with torch.no_grad():
        x = embd(tokens)

        seq_len = x.shape[0]
        head_dim = config['dim'] // config['n_heads']

        for layer in range(config['n_layers']):
            # RMS 1
            rms_1 = torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=model[f"layers.{layer}.attention_norm.weight"], eps=config["norm_eps"])
            
            # GQA
            xq = rms_1 @ torch.transpose(model[f"layers.{layer}.attention.wq.weight"].type(torch.float32), 0, 1)
            xk = rms_1 @ torch.transpose(model[f"layers.{layer}.attention.wk.weight"].type(torch.float32), 0, 1)
            xv = rms_1 @ torch.transpose(model[f"layers.{layer}.attention.wv.weight"].type(torch.float32), 0, 1)
            
            xq = xq.view(seq_len, config['n_heads'], head_dim).transpose(0, 1).contiguous()
            xk = xk.view(seq_len, config['n_kv_heads'], head_dim).repeat_interleave(config['n_heads'] // config['n_kv_heads'], dim=1).transpose(0, 1).contiguous()
            xv = xv.view(seq_len, config['n_kv_heads'], head_dim).repeat_interleave(config['n_heads'] // config['n_kv_heads'], dim=1).transpose(0, 1).contiguous()
            
            xq = apply_rope(xq)
            xk = apply_rope(xk)
            
            logits = (xq @ xk.transpose(-2, -1)) / (head_dim**0.5)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf')),
                diagonal=1
            )
            attn = (logits + mask).softmax(dim=-1) @ xv
            attn_o = attn.transpose(0, 1).reshape(seq_len, -1) @ model[f"layers.{layer}.attention.wo.weight"].type(torch.float32).transpose(0, 1)
            
            # Residuals
            x_2 = x + attn_o
            
            # RMS 2
            rms_2 = torch.nn.functional.rms_norm(x_2, normalized_shape=(x_2.shape[-1],), weight=model[f"layers.{layer}.ffn_norm.weight"], eps=config["norm_eps"])
            
            # FFN
            ffn_w1 = model[f"layers.{layer}.feed_forward.w1.weight"].type(torch.float32)
            ffn_w2 = model[f"layers.{layer}.feed_forward.w2.weight"].type(torch.float32)
            ffn_w3 = model[f"layers.{layer}.feed_forward.w3.weight"].type(torch.float32)
            ffn_output = (torch.nn.functional.silu(rms_2 @ torch.transpose(ffn_w1, 0, 1)) * (rms_2 @ torch.transpose(ffn_w3, 0, 1))) @ torch.transpose(ffn_w2, 0, 1)
            
            # Residuals
            x = x_2 + ffn_output

        rms_f = torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=model["norm.weight"], eps=config["norm_eps"])
        linear_f = rms_f @ torch.transpose(model["output.weight"].type(torch.float32), 0, 1)
        out = torch.nn.functional.softmax(linear_f, dim=-1)[-1]
        values, indices = torch.topk(out, k=1)
        next_token = indices.item()
        print(tokenizer.decode([next_token]))
        tokens = torch.cat((tokens, torch.tensor([next_token])))