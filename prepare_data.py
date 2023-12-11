import argparse
import os
import tiktoken
import numpy as np


def split_train_val(data):
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]
    return train_data, val_data


def tokenize_data(train_data, val_data):
    enc = tiktoken.get_encoding("gpt2")
    tokenized_train = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
    tokenized_val = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)
    return tokenized_train, tokenized_val


def save_train_val_data(data_path: str):
    with open(data_path, "r") as f:
        data = f.read()

    train_data, val_data = split_train_val(data)
    tokenized_train, tokenized_val = tokenize_data(train_data, val_data)

    tokenized_train.tofile(os.path.join(os.path.dirname(__file__), "train.npy"))
    tokenized_val.tofile(os.path.join(os.path.dirname(__file__), "val.npy"))


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

    save_train_val_data(args.data_path)
