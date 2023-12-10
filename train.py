import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from transformer import GPT, GPTConfig


def to_samples_memmap(context_size, memmap_path):
    dataset = np.memmap(memmap_path, dtype=np.uint16, mode="r")
    tokens = len(dataset)
    window_size = context_size + 1
    samples = tokens - window_size + 1

    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches_memmap(batch_size, context_size, memmap_path):
    inputs, targets = to_samples_memmap(context_size, memmap_path)
    s = 0
    while True:
        if s == 0:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]

        batch_inputs = inputs[ids].astype(np.int64)
        batch_targets = targets[ids].astype(np.int64)

        yield batch_inputs, batch_targets
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main():
    train_path = "train.npy"

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

    context_size = config_args["block_size"]
    batch_size = 2
    num_iters = 50000
    steps_per_report = 10

    model = GPT(config)

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    optimizer = optim.SGD(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)

    train_iterator = iterate_batches_memmap(batch_size, context_size, train_path)
    losses = []
    for it, (inputs, targets) in zip(range(num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = loss_and_grad_fn(inputs, targets)
        model.update(optimizer.apply_gradients(grads, model))
        mx.simplify(loss, model.parameters())
        mx.eval(loss, model.parameters())
        losses.append(loss.item())
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            print(f"Iter {it + 1}: Train loss {train_loss:.3f}, ")
            losses = []


if __name__ == "__main__":
    main()
