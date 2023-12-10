import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map


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


model = SmallModel()

mx.eval(model.parameters())

optimizer = optim.SGD(learning_rate=0.01)

inputs = mx.array(np.random.randn(32, 10))
targets = mx.array(np.random.randint(0, 100, 32))

loss_and_grad_fn = nn.value_and_grad(model, model.loss)

grad_accumulation_steps = 4

accumulated_grads = tree_map(lambda x: mx.zeros_like(x), model.parameters())

for step in range(grad_accumulation_steps):
    loss, grads = loss_and_grad_fn(inputs, targets)

    print(f"Step {step}, Current Gradients: {grads}")
    accumulated_grads = tree_map(lambda acc, new: acc + new, accumulated_grads, grads)
    print(f"Step {step}, Accum Gradients: {accumulated_grads}")

normalized_grads = tree_map(lambda x: x / grad_accumulation_steps, accumulated_grads)

optimizer.apply_gradients(accumulated_grads, model)

for param in model.parameters():
    print(param)
