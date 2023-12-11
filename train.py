import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import optim as opt
import math

from mlx.utils import tree_flatten, tree_map
from transformer import GPT, GPTConfig


def create_training_examples(data_path, context_size):
    dataset = np.memmap(data_path, dtype=np.uint16, mode="r")
    tokens = len(dataset)
    window_size = context_size + 1
    samples = tokens - window_size + 1

    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def get_batch(x, y, batch_size):
    s = 0
    while True:
        if s == 0:
            perm = np.random.permutation(x.shape[0])
        ids = perm[s : s + batch_size]

        batch_inputs = x[ids].astype(np.int64)
        batch_targets = y[ids].astype(np.int64)

        yield batch_inputs, batch_targets
        s += batch_size
        if s >= x.shape[0]:
            s = 0


def get_learning_rate(it, lr, min_lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def main(train_path):
    config_args = dict(n_layer=3, n_head=4, n_embd=256)

    config_args["vocab_size"] = 50304
    config_args["block_size"] = 256
    config_args["bias"] = True
    config = GPTConfig(**config_args)

    context_size = config_args["block_size"]
    num_iters = 300
    batch_size = 4
    grad_accumulation_steps = 16
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 0
    lr_decay_iters = 300

    model = GPT(config)

    mx.eval(model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Training a custom GPT model with {nparams / 1e6:.3f} M parameters")

    optimizer = opt.AdamW(learning_rate=max_lr)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)

    x, y = create_training_examples(train_path, context_size)
    train_iterator = get_batch(x, y, batch_size)

    inputs, targets = next(iter(train_iterator))
    inputs, targets = map(mx.array, (inputs, targets))

    _, initial_grads = loss_and_grad_fn(inputs, targets)

    accumulated_grads = tree_map(lambda x: mx.zeros_like(x), model.parameters())
    accumulated_loss = 0.0
    weight_per_step = 1.0 / grad_accumulation_steps
    tic = time.perf_counter()

    total_micro_batch_iters = num_iters * grad_accumulation_steps
    full_iteration_count = 0
    for it in range(total_micro_batch_iters):
        inputs, targets = next(train_iterator)

        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = loss_and_grad_fn(inputs, targets)

        accumulated_grads = tree_map(
            lambda acc, new: acc + new * weight_per_step, accumulated_grads, grads
        )
        accumulated_loss += loss.item()

        if (it + 1) % grad_accumulation_steps == 0:
            curr_lr = get_learning_rate(
                full_iteration_count, max_lr, min_lr, warmup_iters, lr_decay_iters
            )
            optimizer.set_learning_rate(curr_lr)

            model.update(optimizer.apply_gradients(accumulated_grads, model))
            accumulated_grads = tree_map(lambda x: mx.zeros_like(x), model.parameters())
            average_loss = accumulated_loss / grad_accumulation_steps
            accumulated_loss = 0.0

            mx.simplify(loss, model.parameters())
            mx.eval(loss, model.parameters())
            full_iteration_count += 1
            toc = time.perf_counter()
            print(
                f"iter {full_iteration_count}: train loss {average_loss:.3f}, "
                f"it/sec {1.0 / (toc - tic):.3f}, "
                f"lr {curr_lr:.4f}"
            )
            tic = time.perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2-style model on a custom dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="train.npy",
        help="Path to training data *.npy file",
    )

    args = parser.parse_args()

    main(args.data_path)
