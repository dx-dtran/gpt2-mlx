import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 4)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4, 4)

    def __call__(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

    def loss(self, x, target):
        return mx.mean(nn.losses.cross_entropy(self(x), target))


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


model = SmallModel()

mx.eval(model.parameters())

optimizer = optim.SGD(learning_rate=0.01)

inputs = mx.array(np.random.randn(32, 10))
targets = mx.array(np.random.randint(0, 100, 32))

loss_and_grad_fn = nn.value_and_grad(model, model.loss)

grad_accumulation_steps = 4

_, initial_grads = loss_and_grad_fn(inputs, targets)
accumulated_grads = init_accumulator(initial_grads)

for step in range(grad_accumulation_steps):
    loss, grads = loss_and_grad_fn(inputs, targets)

    print(f"Step {step}, Current Gradients: {grads}")
    accumulated_grads = accumulate_grads(accumulated_grads, grads)
    print(f"Step {step}, Accum Gradients: {accumulated_grads}")
    normalized_grads = normalize_grads(accumulated_grads, grad_accumulation_steps)


optimizer.apply_gradients(accumulated_grads, model)

for param in model.parameters():
    print(param)
