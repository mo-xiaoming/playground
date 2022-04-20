import os

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


train_ds = MNIST(root='data/', download=True, train=True, transform=transforms.ToTensor())
test_ds = MNIST(root='data/', download=True, train=False, transform=transforms.ToTensor())
train_ds, val_ds = random_split(train_ds, [50000, 10000])
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size=256)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss_fn=nn.CrossEntropyLoss()


def train(model, loader, optimizer):
    batch_losses = []
    batch_accs = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        batch_losses.append(loss.item())
        batch_accs.append(acc.item())
    return {'loss': torch.tensor(batch_losses).mean(), 'acc': torch.tensor(batch_accs).mean()}


def evaluate(model, loader):
    return train(model, loader, None)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr)
    train_history, val_history = [], []

    for epoch in range(epochs):
        train_metric = train(model, train_loader, optimizer)
        train_history.append(train_metric)

        val_metric = evaluate(model, val_loader)
        val_history.append(val_metric)

        print(f'epoch {epoch}/{epochs}, loss: {train_metric["loss"]:.4f}, acc: {train_metric["acc"]:.4f}, val_loss: {val_metric["loss"]:.4f}, val_acc: {val_metric["acc"]:.4f}')

    return train_history, val_history


class MnistModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self._input_size = input_size
        self._num_classes = num_classes
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.view(-1, self._input_size)
        out = self.linear(xb)
        return out


model = MnistModel(input_size=28*28, num_classes=10)
model.to(device)

model_file = 'simple.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval() # or model.train()
else:
    epochs = 10
    init_metric = evaluate(model, test_loader)
    train_history, val_history = fit(epochs, 0.001, model, train_loader, val_loader)
    final_metric = evaluate(model, test_loader)
    print(f'from {init_metric} to {final_metric}')
    torch.save(model.state_dict(), model_file)

    fig, ax = plt.subplots()
    x = list(range(1, epochs+1))
    ax.plot(x, [i['acc'] for i in train_history], 'm:', label='train acc')
    ax.plot(x, [i['loss'] for i in train_history], 'm', label='train loss')
    ax.plot(x, [i['acc'] for i in val_history], 'b:', label='val acc')
    ax.plot(x, [i['loss'] for i in val_history], 'b', label='val loss')
    ax.plot([0, epochs], [init_metric['acc'], final_metric['acc']], 'k:', label='test acc')
    ax.plot([0, epochs], [init_metric['loss'], final_metric['loss']], 'k', label='test loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.grid()
    fig.tight_layout()
    plt.show()
