import argparse
import numpy as np
import os
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
    parser = argparse.ArgumentParser(description="Convert GPT-2 PyTorch weights to npz")

    parser.add_argument(
        "--weights_path",
        type=str,
        default="pytorch_model.bin",
        help="Path to PyTorch weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="The name of the model",
    )

    args = parser.parse_args()
    state_dict = torch.load(args.weights_path)

    state_dict_transposed = transpose_specific_layers(state_dict)

    input_dir = os.path.dirname(args.weights_path)
    output_path = os.path.join(input_dir, f"{args.model_name}.npz")

    np.savez(
        output_path,
        **{k: v.to(torch.float32).numpy() for k, v in state_dict_transposed.items()},
    )
