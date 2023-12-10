import numpy as np
import torch


def transpose_specific_layers(state_dict):
    layers_to_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key in state_dict.keys():
        if any(key.endswith(suffix) for suffix in layers_to_transpose):
            state_dict[key] = state_dict[key].T
    return state_dict


if __name__ == "__main__":
    state_dict = torch.load("gpt2-xl.bin")

    state_dict_transposed = transpose_specific_layers(state_dict)

    np.savez(
        "gpt2-xl.npz",
        **{k: v.to(torch.float32).numpy() for k, v in state_dict_transposed.items()},
    )
