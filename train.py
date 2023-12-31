import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import optimizer as opt
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
        num_tokens = len(dataset)
        window_size = self.context_size + 1
        examples = num_tokens - window_size + 1

        x = np.lib.stride_tricks.as_strided(
            dataset,
            shape=(examples, window_size),
            strides=(dataset.itemsize, dataset.itemsize),
        )
        return x[:, :-1], x[:, 1:]

    def get_batch_iterator(self, batch_size):
        while True:
            perm = np.random.permutation(self.x.shape[0])

            for start in range(0, self.x.shape[0], batch_size):
                ids = perm[start : start + batch_size]

                batch_inputs = self.x[ids].astype(np.int64)
                batch_targets = self.y[ids].astype(np.int64)

                yield batch_inputs, batch_targets


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

        # init gradient accumulation state
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        self.accumulated_loss = 0.0
        self.iter_num = 0

    def print_parameter_count(self):
        mx.eval(self.model.parameters())
        nparams = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Training a custom GPT model with {nparams / 1e6:.3f} M parameters")

    def print_loss(self, iteration_count, average_loss, tic):
        toc = time.perf_counter()
        print(
            f"iter {iteration_count}: train loss {average_loss:.3f}, "
            f"it/sec {1.0 / (toc - tic):.3f}, "
            f"lr {self.optimizer.learning_rate:.4f}"
        )
        return toc

    def create_data_iterator(self):
        data_loader = DataLoader(self.data_path, self.model_config.block_size)
        return data_loader.get_batch_iterator(self.batch_size)

    def update_learning_rate(self, it):
        if it < self.warmup_iters:
            return self.max_lr * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        new_lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        self.optimizer.set_learning_rate(new_lr)

    def compute_minibatch_loss_grads(self, inputs, targets):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = self.loss_and_grad_fn(inputs, targets)

        self.accumulated_grads = tree_map(
            lambda acc, new: acc + new * (1.0 / self.grad_accumulation_steps),
            self.accumulated_grads,
            grads,
        )

        tree_map(
            lambda grad: mx.eval(grad),
            self.accumulated_grads,
        )

        self.accumulated_loss += loss.item()
        return loss

    def compute_batch_loss(self, loss):
        average_loss = self.accumulated_loss / self.grad_accumulation_steps
        self.accumulated_loss = 0.0
        mx.simplify(loss, self.model.parameters())
        mx.eval(loss, self.model.parameters())
        return average_loss

    def perform_gradient_step(self):
        self.model.update(
            self.optimizer.apply_gradients(self.accumulated_grads, self.model)
        )
        self.accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )

    def train(self):
        self.print_parameter_count()

        train_data = self.create_data_iterator()
        tic = time.perf_counter()

        for iteration in range(self.num_iters * self.grad_accumulation_steps):
            inputs, targets = next(train_data)
            loss = self.compute_minibatch_loss_grads(inputs, targets)

            if (iteration + 1) % self.grad_accumulation_steps == 0:
                self.perform_gradient_step()
                self.update_learning_rate(self.iter_num)
                batch_loss = self.compute_batch_loss(loss)
                tic = self.print_loss(self.iter_num, batch_loss, tic)
                self.iter_num += 1


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
        num_iters=1000,
        batch_size=12,
        grad_accumulation_steps=4,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_iters=200,
        lr_decay_iters=1000,
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
