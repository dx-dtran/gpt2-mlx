import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import optim as opt
import math

from mlx.utils import tree_flatten, tree_map
from transformer import GPT, GPTConfig
from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_iters: int
    batch_size: int
    grad_accumulation_steps: int
    max_lr: float
    min_lr: float
    warmup_iters: int
    lr_decay_iters: int


class DataLoader:
    def __init__(self, data_path, context_size):
        self.data_path = data_path
        self.context_size = context_size
        self.x, self.y = self.create_training_examples()

    def create_training_examples(self):
        dataset = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        tokens = len(dataset)
        window_size = self.context_size + 1
        samples = tokens - window_size + 1

        X = np.lib.stride_tricks.as_strided(
            dataset,
            shape=(samples, window_size),
            strides=(dataset.itemsize, dataset.itemsize),
        )
        return X[:, :-1], X[:, 1:]

    def get_batch_iterator(self, batch_size):
        s = 0
        while True:
            if s == 0:
                perm = np.random.permutation(self.x.shape[0])
            ids = perm[s : s + batch_size]

            batch_inputs = self.x[ids].astype(np.int64)
            batch_targets = self.y[ids].astype(np.int64)

            yield batch_inputs, batch_targets
            s += batch_size
            if s >= self.x.shape[0]:
                s = 0


class GPTTrainer:
    def __init__(self, data_path, train_config: TrainConfig, model_config: GPTConfig):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path, model_config.block_size)

        # training hyperparameters
        self.train_config = train_config
        self.batch_size = train_config.batch_size
        self.num_iters = train_config.num_iters
        self.grad_accumulation_steps = train_config.grad_accumulation_steps
        self.max_lr = train_config.max_lr
        self.min_lr = train_config.min_lr
        self.warmup_iters = train_config.warmup_iters
        self.lr_decay_iters = train_config.lr_decay_iters

        # initialize the model and optimizer
        self.model_config = model_config
        self.model = GPT(model_config)
        self.optimizer = opt.AdamW(learning_rate=self.max_lr)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.model.loss)

    def print_model_parameters(self):
        mx.eval(self.model.parameters())
        nparams = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Training a custom GPT model with {nparams / 1e6:.3f} M parameters")

    def print_progress(self, iteration_count, average_loss, tic):
        toc = time.perf_counter()
        print(
            f"iter {iteration_count}: train loss {average_loss:.3f}, "
            f"it/sec {1.0 / (toc - tic):.3f}, "
            f"lr {self.optimizer.learning_rate:.4f}"
        )
        return toc

    def get_learning_rate(self, it):
        if it < self.warmup_iters:
            return self.max_lr * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def update_learning_rate(self, iteration_count):
        current_lr = self.get_learning_rate(iteration_count)
        self.optimizer.set_learning_rate(current_lr)

    def process_batch(self, inputs, targets):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = self.loss_and_grad_fn(inputs, targets)
        return loss, grads

    def train(self):
        self.print_model_parameters()

        data_loader = DataLoader(self.data_path, self.model_config.block_size)
        train_iterator = data_loader.get_batch_iterator(self.train_config.batch_size)

        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        accumulated_loss = 0.0
        weight_per_step = 1.0 / self.train_config.grad_accumulation_steps
        tic = time.perf_counter()

        total_micro_batch_iters = (
            self.train_config.num_iters * self.train_config.grad_accumulation_steps
        )
        full_iteration_count = 0
        for it in range(total_micro_batch_iters):
            inputs, targets = next(train_iterator)
            loss, grads = self.process_batch(inputs, targets)

            accumulated_grads = tree_map(
                lambda acc, new: acc + new * weight_per_step,
                accumulated_grads,
                grads,
            )
            accumulated_loss += loss.item()

            if (it + 1) % self.train_config.grad_accumulation_steps == 0:
                self.update_learning_rate(full_iteration_count)
                self.model.update(
                    self.optimizer.apply_gradients(accumulated_grads, self.model)
                )
                accumulated_grads = tree_map(
                    lambda x: mx.zeros_like(x), self.model.parameters()
                )
                average_loss = (
                    accumulated_loss / self.train_config.grad_accumulation_steps
                )
                accumulated_loss = 0.0

                mx.simplify(loss, self.model.parameters())
                mx.eval(loss, self.model.parameters())
                full_iteration_count += 1
                tic = self.print_progress(full_iteration_count, average_loss, tic)


def main(train_path):
    model_config = GPTConfig(
        n_layer=3,
        n_head=4,
        n_embd=256,
        vocab_size=50304,
        block_size=256,
        bias=True,
    )

    train_config = TrainConfig(
        num_iters=300,
        batch_size=4,
        grad_accumulation_steps=16,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_iters=0,
        lr_decay_iters=300,
    )
    trainer = GPTTrainer(train_path, train_config, model_config)
    trainer.train()


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
