import tiktoken
import mlx.core as mx

from mlx.utils import tree_unflatten
from transformer import GPT, GPTConfig


def load_model(model_path):
    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
    }["gpt2"]

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
    x = mx.expand_dims(mx.array(start_ids, dtype=mx.uint32), axis=0)

    for k in range(10):
        y = model.generate(
            x,
            256,
            temperature=0.8,
        )
        result = y[0].tolist()
        print(decode(result))
        print("---------------")


if __name__ == "__main__":
    model = load_model("gpt2.npz")

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    sample("a story about a boy in a castle", model, encode, decode)
