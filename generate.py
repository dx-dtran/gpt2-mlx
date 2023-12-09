import tiktoken
import mlx.core as mx
import time

from mlx.utils import tree_unflatten
from transformer import GPT, GPTConfig


def load_model(model_path, model_name="gpt2-xl"):
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

    weights = mx.load(model_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    return model


def sample(prompt, model, encode, decode):
    start_ids = encode(prompt)

    # try batch inference
    x = mx.array([start_ids], dtype=mx.uint32)

    tokens = []
    start = time.time()
    for token in model.generate(x, max_new_tokens=256):
        tokens.append(token.item())
    end = time.time()
    print(prompt + decode(tokens))
    print("---------------")
    print(
        f"Time: {end - start:.3f} s, Tokens per second: {len(tokens) / (end - start)}"
    )
    print("---------------")


if __name__ == "__main__":
    model = load_model("gpt2.npz", "gpt2")

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    sample(
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        model,
        encode,
        decode,
    )
