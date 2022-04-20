import random

import pandas as pd
from matplotlib import pyplot as plt
import math
import time
import os
import numpy as np
import torch
from torch.utils import data
from torch.distributions import multinomial
from torch import nn
import torchvision
from torchvision import transforms


#%%


def _handle_csv():
    data_file = os.path.join('data', 'house_tiny.csv')
    if not os.path.exists(data_file):
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        with open(data_file, 'w') as f:
            f.write('NumRooms,Alley,Price\n')  # Column names
            f.write('NA,Pave,127500\n')  # Each row represents a data example
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')

    data = pd.read_csv(data_file)
    print(data)

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X, y)

#%%


def _limit():
    def f(x):
        return 3 * x ** 2 - 4 * x


    def numerical_lim(f, x, h):
        return (f(x + h) - f(x)) / h


    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
        h *= 0.1

#%%


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()

    def has_one_axis(X):
        if hasattr(X, 'ndim') and X.ndim == 1:
            return True
        if isinstance(X, list) and not hasattr(X[0], '__len__'):
            return True
        return False

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()

#%%


def _draw_f():
    def f(x):
        return 3 * x ** 2 - 4 * x

    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

#%%


def _grad():
    x = torch.arange(4.0, requires_grad=True)
    print(x.grad)

    y = 2 * torch.dot(x, x) # 2x^2 ?
    y.backward()
    print(x.grad == 4 * x)

    x.grad.zero_() # important
    y = x.sum()
    y.backward()
    print(x.grad == torch.ones(4))

    x.grad.zero_()
    y = x * x
    y.sum().backward() # or y.backward(torch.ones(len(x)))
    print(x.grad == torch.tensor([0, 2, 4, 6]))

#%%


def _draw_die():
    fair_probs = torch.ones(6)/ 6
    counts = multinomial.Multinomial(10, fair_probs).sample((500,))
    cum_counts = counts.cumsum(dim=0)
    estimates = cum_counts/cum_counts.sum(dim=1, keepdims=True)

    set_figsize((6, 4.5))
    for i in range(6):
        plt.plot(estimates[:, i].numpy(), label=(f'P(die={i+1})'))
    plt.axhline(y=0.167, color='black', linestyle='dashed')
    plt.gca().set_xlabel('Groups of experiments')
    plt.gca().set_ylabel('Estimated probability')
    plt.legend()
    plt.show()

#%%


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times)/ len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        print(np.array(self.times))
        print(np.array(self.times).cumsum())
        return np.array(self.times).cumsum().tolist()


def _speed():
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)

    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')

#%%


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma **2 * (x - mu) ** 2)


def _draw_norm():
    x = np.arange(-7, 7, 0.01)
    params = [(0, 1), (0, 2), (3, 1)]
    plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5), legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

#%%


def synthetic_data(w, b, num_exmaples):
    '''Generate y = Xw +b + noise'''
    X = torch.normal(0, 1, (num_exmaples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

#%%


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squired_loss(y_hat, y):
    return (y_hat - y.view(y_hat.shape)) ** 2 /2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def _draw_wb():
    set_figsize()
    plt.scatter(features[:, 0], labels, 1)
    plt.show()

#%%


def _linreg_from_scratch():
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squired_loss
    batch_size = 10

    w = torch.normal(0.0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    true_w, true_b = torch.tensor([2, -3.4]), 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
    print(f'error in estimating b: {true_b - b}')

#%%


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def _linreg_concise():
    true_w, true_b = torch.tensor([2, -3.4]), 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10

    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))

    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    for epoch in range(3):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('error in estimating w:', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('error in estimating b:', true_b - b)

#%%


def _something():
    def get_fashion_mnist_labels(labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
        figsize = (num_cols * scale, num_rows * scale)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                ax.imshow(img.numpy())
            else:
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        fig.tight_layout()
        plt.show()


    def _show_mnist():
        trans = transforms.ToTensor()
        mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, transform=trans, download=True)

        X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
        show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


    def load_data_fashion_mnist(batch_size, resize=None):
        def get_dataloader_workers():
            return 4

        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition


    def net(X):
        return softmax(torch.matmul(X.view(-1, W.shape[0]), W) + b)


    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])


    def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())


    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=64)

    num_inputs = 28*28
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)


    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

#%%


import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))