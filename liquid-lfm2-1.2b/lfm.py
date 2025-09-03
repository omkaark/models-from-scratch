from pathlib import Path
import torch
import json
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file

tokenizer_path = "./model"
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

with open("./model/config.json", "r") as f:
    config = json.load(f)

model = load_file("./model/model.safetensors", device='cpu')
embd = torch.nn.Embedding(config['vocab_size'], config['hidden_size'])
embd.load_state_dict({'weight': model[f'model.embed_tokens.weight']})

messages = [
    {"role": "system", "content": 'Follow the instructions.'},
    {"role": "user",   "content": "Explain the duality of man."}
]
tokens = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True
)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]

def apply_rope(x: torch.Tensor):
    _, H, S, D =  x.shape
    freqs = 1 / config['rope_theta'] ** (torch.arange(0, D, 2) / D)
    freqs_per_token = torch.outer(torch.arange(S), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_per_token), freqs_per_token)
    
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

layer_cache = [{} for i in range(config['num_hidden_layers'])]

while tokens[-1] != 7:

    x = embd(tokens).unsqueeze(0)

    for layer in range(config['num_hidden_layers']):
        S = x.shape[1]

        x_norm = torch.nn.functional.rms_norm(x, normalized_shape=(config['hidden_size'],), weight=model[f'model.layers.{layer}.operator_norm.weight'], eps=config['norm_eps']).type(torch.bfloat16)

        if layer in config['full_attn_idxs']:  # Attention
            xq = x_norm @ model[f'model.layers.{layer}.self_attn.q_proj.weight'].T
            xk = x_norm @ model[f'model.layers.{layer}.self_attn.k_proj.weight'].T
            xv = x_norm @ model[f'model.layers.{layer}.self_attn.v_proj.weight'].T

            xq = xq.view(1, S, config['num_attention_heads'], -1).transpose(-2, -3)
            xk = xk.view(1, S, config['num_key_value_heads'], -1).repeat_interleave(config['num_attention_heads'] // config['num_key_value_heads'], dim=2).transpose(-2, -3)
            xv = xv.view(1, S, config['num_key_value_heads'], -1).repeat_interleave(config['num_attention_heads'] // config['num_key_value_heads'], dim=2).transpose(-2, -3)

            xq = apply_rope(xq.type(torch.float32))
            xk = apply_rope(xk.type(torch.float32))

            xq = torch.nn.functional.rms_norm(xq, normalized_shape=(xq.shape[-1],), weight=model[f'model.layers.{layer}.self_attn.q_layernorm.weight'], eps=config['norm_eps'])
            xk = torch.nn.functional.rms_norm(xk, normalized_shape=(xk.shape[-1],), weight=model[f'model.layers.{layer}.self_attn.k_layernorm.weight'], eps=config['norm_eps'])
            
            score = ((xq @ xk.transpose(-1, -2)) / (xk.shape[-1] ** 0.5)).type(torch.float32) + torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
            attn = torch.softmax(score, dim=-1) @ xv.float()
            x_operator = attn.to(torch.bfloat16).transpose(1, 2).reshape(1, S, -1) @ model[f'model.layers.{layer}.self_attn.out_proj.weight'].T
        else: # Conv layer
            # (1, S, D) @ (1, D, 3D) = (1, S, 3D) -> T -> (1, 3D, S)
            BCx = (x_norm @ model[f'model.layers.{layer}.conv.in_proj.weight'].T).transpose(-1, -2)

            # (1, 3D, S) -> (1, D, S), (1, D, S), (1, D, S)
            B, C, x_c = BCx.chunk(3, dim=-2)

            # (1, D, S) * (1, D, S) -> (1, D, S)
            x_c = B * x_c

            # (1, D, S) conv (D, 1, 3) -> (1, D, S + 2)
            x_c = torch.nn.functional.conv1d(
                x_c,
                weight=model[f'model.layers.{layer}.conv.conv.weight'],
                padding=config['conv_L_cache'] - 1,
                groups=config['hidden_size'],
            )

            # (1, D, S) * (1, D, S) -> (1, D, S)
            x_c = C * x_c[:, :, :S]

            # (1, S, D) @ (D, D) -> (1, S, D)
            x_operator = x_c.transpose(-1, -2) @ model[f'model.layers.{layer}.conv.out_proj.weight'].T

        x = x + x_operator

        x_norm = torch.nn.functional.rms_norm(x, normalized_shape=(config['hidden_size'],), weight=model[f'model.layers.{layer}.ffn_norm.weight'], eps=config['norm_eps']).type(torch.bfloat16)

        ffn_w1 = model[f"model.layers.{layer}.feed_forward.w1.weight"].type(torch.bfloat16)
        ffn_w2 = model[f"model.layers.{layer}.feed_forward.w2.weight"].type(torch.bfloat16)
        ffn_w3 = model[f"model.layers.{layer}.feed_forward.w3.weight"].type(torch.bfloat16)
        ffn_o = (torch.nn.functional.silu(x_norm @ torch.transpose(ffn_w1, 0, 1)) * (x_norm @ torch.transpose(ffn_w3, 0, 1))) @ torch.transpose(ffn_w2, 0, 1)

        x = x + ffn_o

    x = torch.nn.functional.rms_norm(x, normalized_shape=(config['hidden_size'],), weight=model['model.embedding_norm.weight'], eps=config['norm_eps']).type(torch.bfloat16)
    out = x @ model[f'model.embed_tokens.weight'].T

    out_softmax = torch.nn.functional.softmax(out[:, -1, :].float(), dim=-1)
    values, indices = torch.topk(out_softmax, k=1)
    tokens = torch.cat((tokens, torch.tensor([indices])), dim=-1)
    print(tokenizer.decode(indices[0]), end='', flush=True)
