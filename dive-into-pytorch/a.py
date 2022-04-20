import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch2trt import torch2trt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


inputs = torch.tensor([
    [73, 67, 43],
    [91, 88, 64],
    [87, 134, 58],
    [102, 43, 37],
    [69, 96, 70]
    ], dtype=torch.float32)

targets = torch.tensor([
    [56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119]
    ], dtype=torch.float32)

batch_size = 5
lr = 1e-5

train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

class SimplestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, xb):
        return self.linear(xb)


model = SimplestModel().to(device)
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=lr)


def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


fit(100, model, loss_fn, opt, train_dl)

model.eval()
torch.save(model, 'a_model.pth')

#x = torch.zeros(1, 3, device='cuda')
x = torch.tensor([[74., 68., 44.]], device='cuda')
print(x)

torch.onnx.export(torch.load('a_model.pth'), x, 'a_model.onnx', do_constant_folding=True, input_names=['input1'], output_names=['output1'])

model_trt = torch2trt(torch.load('a_model.pth'), [x], max_batch_size=5, fp16_mode=True)

print(model(x))
print(model_trt(x))

with open('a_model.engine', 'wb') as f:
    f.write(model_trt.engine.serialize())
