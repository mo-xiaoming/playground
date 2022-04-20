import numpy as np
import matplotlib.pyplot as plt


def create_axis(ax, alpha, beta):
    x = np.arange(10, dtype=float)
    y = alpha + x * beta

    ax.plot(x, y)


def create_plot():
    alpha = 3.

    fig, axes = plt.subplots(1, 4, figsize=(5, 4))

    for i, ax in enumerate(axes):
        create_axis(ax, alpha, i)


if __name__ == '__main__':
    create_plot()
plt.show()
