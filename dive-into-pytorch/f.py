import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotate(),
    # transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)])
val_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

def loader_2(root, batch_size, train_transforms, val_transforms):
    train_ds = ImageFolder(root=os.path.join(root, 'train'), transform=train_transforms)
    val_ds = ImageFolder(root=os.path.join(root, 'test'), transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, num_workers=4, pin_memory=True)

    return train_loader, val_loader


train_loader, val_loader = loader_2('data/cifar10', batch_size=64, train_transforms=train_tfms, val_transforms=val_tfms)
print(f'train: {len(train_loader)} batches, val: {len(val_loader)} batches')


def _work(model, loader, optimizer, scheduler, grad_clip, loss_fn=nn.CrossEntropyLoss()):
    batch_losses = []
    batch_accs = []
    lrs = []
    dataset_start = time.time()
    for i, ds in enumerate(loader, 1):
        images, labels = ds[0].to(device), ds[1].to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        if optimizer:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        batch_losses.append(loss.item())
        batch_accs.append(acc.item())

        total_dataset_time = time.time() - dataset_start
        if i % 5 == 0:
            print(f'\r    batch [{i}/{len(loader)}] {total_dataset_time:.2f} secs, ETA {total_dataset_time/i*(len(loader)-i):.2f} secs', end='')
    return {'loss': torch.tensor(batch_losses).mean(), 'acc': torch.tensor(batch_accs).mean(), 'lrs': lrs}


def train(model, loader, optimizer, scheduler, grad_clip):
    model.train()
    return _work(model, loader, optimizer, scheduler, grad_clip)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    return _work(model, loader, optimizer=None, scheduler=None, grad_clip=None)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, train_loader, val_loader, max_lr, weight_decay, grad_clip=None, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    train_history, val_history = [], []

    for epoch in range(epochs):
        start = time.time()

        train_metric = train(model, train_loader, optimizer, sched, grad_clip)
        train_history.append(train_metric)

        val_metric = evaluate(model, val_loader)
        val_history.append(val_metric)

        print(f'\repoch {epoch+1}/{epochs}, {time.time()-start:.2f} secs, last_lr: {train_metric["lrs"][-1]:.6f}, loss: {train_metric["loss"]:.4f}, acc: {train_metric["acc"]:.4f}, val_loss: {val_metric["loss"]:.4f}, val_acc: {val_metric["acc"]:.4f}')

    return train_history, val_history


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
                nn.MaxPool2d(4),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


model = ResNet9(3, 10)
model.to(device)

model_file = 'simple.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval() # or model.train()
else:
    epochs = 10
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    init_val_metric = evaluate(model, val_loader)
    print('\rinitial val metric collected          ')
    training_start = time.time()
    train_history, val_history = fit(epochs, model, train_loader, val_loader, max_lr=max_lr, weight_decay=weight_decay, grad_clip=grad_clip)
    print(f'training took {time.time()-training_start:.2f} seconds')
    #torch.save(model.state_dict(), model_file)

    fig, ax = plt.subplots()
    train_x = range(1, len(train_history)+1)
    ax.plot(train_x, [i['acc'] for i in train_history], 'm', label='train acc')
    ax.plot(train_x, [i['loss'] for i in train_history], 'm:', label='train loss')
    val_x = range(0, len(val_history)+1)
    ax.plot(val_x, [init_val_metric['acc']] + [i['acc'] for i in val_history], 'b', label='val acc')
    ax.plot(val_x, [init_val_metric['loss']] + [i['loss'] for i in val_history], 'b:', label='val loss')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.grid()
    fig.tight_layout()
    plt.show()
