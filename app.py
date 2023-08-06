import torch
import numpy
import pandas
import torch.nn as neuralNetwork
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
needTraining = False
data = pandas.read_csv("test_data.csv")
input = torch.from_numpy(data[['x','y']].to_numpy(dtype='float32'))
output = torch.from_numpy(data['z'].to_numpy(dtype='float32'))
model =neuralNetwork.Linear(2,1,bias=True)
if(not needTraining):
  model.load_state_dict(torch.load("trained_model.txt"))

optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
loss_fn = F.mse_loss
train_ds = TensorDataset(input,output)
train_dl = DataLoader(dataset=train_ds,batch_size=128,shuffle=True)

def fit(num_epochs,model,loss_fn,opt,dataloader):
  for epoch in range(num_epochs):
    for xb,yb in dataloader:
      pred = model(xb).flatten()
      loss =loss_fn(pred,yb)
      loss.backward()
      opt.step()
      opt.zero_grad()

if(needTraining):
  fit(100,model=model,loss_fn=loss_fn,opt=optimizer,dataloader=train_dl)
  torch.save(model.state_dict(),"trained_model.txt")      
input_x = 1.
input_y = 3.
preds = model(torch.tensor([input_x,input_y],dtype=torch.float32)).flatten()
print(round(preds.item()))