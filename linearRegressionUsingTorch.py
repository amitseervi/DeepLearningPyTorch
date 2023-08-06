import torch.nn as neuralNetwork
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy
import torch

inputs = torch.from_numpy(numpy.array(
    [
        [73., 67., 43.],
        [91, 88, 64.],
        [87, 134, 58.],
        [102, 43, 37.],
        [69, 96, 70.],
        [73, 67, 43.],
        [91, 88, 64.],
        [87, 134, 58.],
        [102, 43, 37],
        [69, 96, 70],
        [73, 67, 43],
        [91, 88, 64],
        [87, 134, 58],
        [102, 43, 37],
        [69, 96, 70],
    ]
))

targets = torch.from_numpy(numpy.array(
    [
        [56.0, 70],
        [81, 101],
        [119, 133],
        [22, 37],
        [103, 119],
        [56, 70],
        [81, 101],
        [119, 133],
        [22, 37],
        [103, 119],
        [56, 70],
        [81, 101],
        [119, 133],
        [22, 37],
        [103, 119],
    ]
))


def fit(train_dl, num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()


def evaluate(model, loss_fn):
    preds = model(inputs)
    loss = loss_fn(targets, preds)
    print(loss)


if __name__ == '__main__':
    train_ds = TensorDataset(inputs, targets)
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = neuralNetwork.Linear(3, 2, dtype=torch.float64)
    loss_fn = F.mse_loss
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    fit(train_dl=train_dl, num_epochs=200, model=model, loss_fn=loss_fn, opt=opt)
    evaluate(model=model, loss_fn=loss_fn)
