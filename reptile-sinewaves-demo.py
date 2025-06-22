"""Original script from: https://gist.github.com/joschu/f503500cda64f2ce87c8288906b09e2d#file-reptile-sinewaves-demo-py"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd as ag
from torch import nn

seed = 0
plot = True
innerstepsize = 0.02  # stepsize in inner SGD
innerepochs = 1  # number of epochs of each inner SGD
outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization
niterations = (
    30000  # number of outer updates; each iteration we sample one task and update on it
)

# Create assets directory for saving plots
os.makedirs("assets/reptile_history", exist_ok=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

rng = np.random.RandomState(seed)
torch.manual_seed(seed)

# Define task distribution
x_all = np.linspace(-5, 5, 50)[:, None]  # All of the x points
ntrain = 10  # Size of training minibatches


def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2 * np.pi)
    ampl = rng.uniform(0.1, 5)

    def f_randomsine(x):
        return np.sin(x + phase) * ampl

    return f_randomsine


# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
).to(device)  # Move model to GPU


def totorch(x):
    return ag.Variable(torch.Tensor(x).to(device))  # Move tensor to GPU


def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.data -= innerstepsize * param.grad.data


def predict(x):
    x = totorch(x)
    return model(x).cpu().data.numpy()  # Move result back to CPU for numpy conversion


# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]

# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(innerepochs):
        for start in range(0, len(x_all), ntrain):
            mbinds = inds[start : start + ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations)  # linear schedule
    model.load_state_dict(
        {
            name: weights_before[name]
            + (weights_after[name] - weights_before[name]) * outerstepsize
            for name in weights_before
        }
    )

    # Periodically plot the results on a particular task and minibatch
    if plot and iteration == 0 or (iteration + 1) % 1000 == 0:
        plt.figure(figsize=(10, 6))
        f = f_plot
        weights_before = deepcopy(model.state_dict())  # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0, 0, 1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter + 1) % 8 == 0:
                frac = (inneriter + 1) / 32
                plt.plot(
                    x_all,
                    predict(x_all),
                    label="pred after %i" % (inneriter + 1),
                    color=(frac, 0, 1 - frac),
                )
        plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4, 4)
        plt.legend(loc="lower right")
        plt.title(f"Reptile Training - Iteration {iteration + 1}")
        plt.xlabel("x")
        plt.ylabel("y")

        # Save the plot as PNG
        filename = f"assets/reptile_history/iteration_{iteration + 1:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

        model.load_state_dict(weights_before)  # restore from snapshot
        print("-----------------------------")
        print(f"iteration               {iteration + 1}")
        print(f"loss on plotted curve   {lossval:.3f}")
        print(f"plot saved to           {filename}")
