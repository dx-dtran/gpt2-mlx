import numpy as np
import torch


if __name__ == "__main__":
    state_dict = torch.load("gpt2-xl.bin")
    for key, value in state_dict.items():
        print(f"Layer: {key}")
        print(f"Shape: {value.shape}")
    np.savez(
        "gpt2-xl.npz",
        **{k: v.to(torch.float32).numpy() for k, v in state_dict.items()},
    )
