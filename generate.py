import argparse
import tiktoken
import time
import mlx.core as mx

from mlx.utils import tree_unflatten, tree_flatten
from transformer import GPT, GPTConfig


def load_model(model_name):
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

    model = GPT(config)

    weights = mx.load(model_name + ".npz")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")

    return model


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
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        help="The prompt to generate text from",
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2-xl", help="The name of the model to use"
    )

    args = parser.parse_args()

    model = load_model(args.model_name)
    generate_text(args.prompt, model)
