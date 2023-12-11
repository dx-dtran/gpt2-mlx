import argparse
import os
import tiktoken
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data for custom GPT-2-style model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="train.txt",
        help="Path to training data file",
    )

    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.npy"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.npy"))
