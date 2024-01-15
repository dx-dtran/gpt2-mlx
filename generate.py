import argparse
import tiktoken
import time
import mlx.core as mx
import os
import json

from mlx.utils import tree_unflatten, tree_flatten
from transformer import GPT, GPTConfig


def load_weights(gpt_model, weights):
    gpt_model.update(tree_unflatten(list(weights.items())))
    mx.eval(gpt_model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(gpt_model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")


def load_custom_model(checkpoint_dir):
    model_weights_path = os.path.join(checkpoint_dir, "model_weights.npz")
    model_config_path = os.path.join(checkpoint_dir, "model_config.json")

    with open(model_config_path, "r") as f:
        config_args = json.load(f)

    config = GPTConfig(**config_args)

    gpt_model = GPT(config)

    weights = mx.load(model_weights_path)
    load_weights(gpt_model, weights)

    return gpt_model


def load_openai_model(model_name):
    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_name]

    config_args["vocab_size"] = 50257
    config_args["block_size"] = 1024
    config_args["bias"] = True

    config = GPTConfig(**config_args)

    gpt_model = GPT(config)

    weights = mx.load(model_name + ".npz")
    load_weights(gpt_model, weights)

    return gpt_model


def generate_text(prompt: str, model: GPT):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    start_ids = encode(prompt)

    x = mx.array([start_ids], dtype=mx.uint32)

    print(prompt, end="")
    tokens = []
    start = time.time()
    for token in model.generate(x, max_new_tokens=256):
        tok = token.item()
        tokens.append(tok)
        print(decode([tok]), end="", flush=True)
    end = time.time()
    print("---------------")
    print(
        f"time: {end - start:.3f} s, tokens per second: {len(tokens) / (end - start)}"
    )
    print("---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from GPT-2")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_name", type=str, help="The name of a pre-trained GPT-2 model to use"
    )
    group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory of a custom model",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously",
        help="The prompt to generate text from",
    )

    args = parser.parse_args()

    if args.model_name:
        model = load_openai_model(args.model_name)
        generate_text(args.prompt, model)

    elif args.checkpoint_dir:
        model = load_custom_model(args.checkpoint_dir)
        generate_text(args.prompt, model)
