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


def init_accumulator(grads_structure):
    if isinstance(grads_structure, dict):
        return {k: init_accumulator(v) for k, v in grads_structure.items()}
    elif isinstance(grads_structure, list):
        return [init_accumulator(item) for item in grads_structure]
    else:
        return mx.zeros_like(grads_structure)


def accumulate_grads(accumulated, new):
    if isinstance(new, dict):
        return {k: accumulate_grads(accumulated[k], v) for k, v in new.items()}
    elif isinstance(new, list):
        return [accumulate_grads(acc, n) for acc, n in zip(accumulated, new)]
    else:
        return accumulated + new


def normalize_grads(accumulated, grad_accumulation_steps):
    if isinstance(accumulated, dict):
        return {
            k: normalize_grads(v, grad_accumulation_steps)
            for k, v in accumulated.items()
        }
    elif isinstance(accumulated, list):
        return [normalize_grads(item, grad_accumulation_steps) for item in accumulated]
    else:
        return accumulated / grad_accumulation_steps


def main():
    train_path = "train.npy"

    config_args = {
        "gpt2": dict(n_layer=1, n_head=12, n_embd=768),
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
    grad_accumulation_steps = 8

    model = GPT(config)

    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)

    train_iterator = iterate_batches_memmap(batch_size, context_size, train_path)
    losses = []

    inputs, targets = next(iter(train_iterator))
    inputs, targets = map(mx.array, (inputs, targets))

    _, initial_grads = loss_and_grad_fn(inputs, targets)

    accumulated_grads = init_accumulator(initial_grads)

    for it, (inputs, targets) in zip(range(num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = loss_and_grad_fn(inputs, targets)

        accumulated_grads = accumulate_grads(accumulated_grads, grads)

        if (it + 1) % grad_accumulation_steps == 0:
            normalized_grads = normalize_grads(
                accumulated_grads, grad_accumulation_steps
            )
            model.update(optimizer.apply_gradients(normalized_grads, model))
            accumulated_grads = init_accumulator(accumulated_grads)

        mx.simplify(loss, model.parameters())
        mx.eval(loss, model.parameters())
        losses.append(loss.item())
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            print(f"Iter {it + 1}: Train loss {train_loss:.3f}, ")
            losses = []


if __name__ == "__main__":
    main()
