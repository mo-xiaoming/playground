import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz

def loader_3(root, batch_size):
    train_ds = ImageFolder(root=os.path.join(root, 'train'), transform=transforms.ToTensor())
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - 5000, 5000])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, pin_memory=True)

    test_ds = ImageFolder(root=os.path.join(root, 'test'), transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=True)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = loader_3('data/cifar10', batch_size=256)
print(f'train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss_fn=nn.CrossEntropyLoss()


def train(model, loader, optimizer):
    if optimizer:
        model.train()
    else:
        model.eval()

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
    with torch.no_grad():
        return train(model, loader, None)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr)
    train_history, val_history = [], []

    for epoch in range(epochs):
        train_metric = train(model, train_loader, optimizer)
        train_history.append(train_metric)

        val_metric = evaluate(model, val_loader)
        val_history.append(val_metric)

        print(f'epoch {epoch+1}/{epochs}, loss: {train_metric["loss"]:.4f}, acc: {train_metric["acc"]:.4f}, val_loss: {val_metric["loss"]:.4f}, val_acc: {val_metric["acc"]:.4f}')

    return train_history, val_history


class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

                nn.Flatten(),
                nn.Linear(256*4*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
                )

    def forward(self, xb):
        return self.network(xb)


model = CifarModel()
model.to(device)

model_file = 'simple.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval() # or model.train()
else:
    epochs = 3
    init_train_metric = evaluate(model, train_loader)
    init_val_metric = evaluate(model, val_loader)
    train_history1, val_history1 = fit(epochs, 0.001, model, train_loader, val_loader)
    train_history2, val_history2 = fit(epochs, 0.0001, model, train_loader, val_loader)
    train_history3, val_history3 = fit(epochs, 0.00001, model, train_loader, val_loader)
    test_metric = evaluate(model, test_loader)
    print(test_metric)
    #torch.save(model.state_dict(), model_file)

    train_history = train_history1 + train_history2 + train_history3
    val_history = val_history1 + val_history2 + val_history3

    fig, ax = plt.subplots()
    x = list(range(0, len(train_history)+1))
    ax.plot(x, [init_train_metric['acc']] + [i['acc'] for i in train_history], 'm:', label='train acc')
    ax.plot(x, [init_train_metric['loss']] + [i['loss'] for i in train_history], 'm', label='train loss')
    ax.plot(x, [init_val_metric['acc']] + [i['acc'] for i in val_history], 'b:', label='val acc')
    ax.plot(x, [init_val_metric['loss']] + [i['loss'] for i in val_history], 'b', label='val loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.grid()
    fig.tight_layout()
    plt.show()
