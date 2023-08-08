import torch
import numpy
import pandas
from torch import Tensor
import torchvision
import torchvision.transforms as visionTransform
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os

dataset = MNIST('./data',download=False,transform=visionTransform.ToTensor())
test_dataset = MNIST('./data',train=False,transform=visionTransform.ToTensor())

train_ds,val_ds = random_split(dataset=dataset,lengths=[50000,10000])
train_dl = DataLoader(train_ds,batch_size=128,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=128,shuffle=False)

input_size=28*28
output_size=10

class MnistModel(nn.Module):
    def __init__(self, input_size:int,output_size:int,lossFn,accuracyFn) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        self.lossFn =lossFn
        self.accuracyFn = accuracyFn
    
    def forward(self,inputBatch:Tensor):
        inputBatch=inputBatch.reshape(-1,784)
        out = self.linear(inputBatch)
        out = F.softmax(out,dim=1)
        return out
    
    def training_step(self,batch):
        images,labels = batch
        preds = self(images)
        loss = self.lossFn(preds,labels)
        return loss
    
    def validation_step(self,batch):
        images,labels = batch
        preds = self(images)
        loss = self.lossFn(preds,labels)
        acc = self.accuracyFn(preds,labels)
        return {
            'val_loss':loss,
            'val_acc':acc
        }
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_acc = [x['val_acc'] for x in outputs]
        avg_loss = torch.stack(batch_losses).mean()
        avg_acc = torch.stack(batch_acc).mean()
        return {'avg_loss':avg_loss.item(),'avg_acc':avg_acc.item()}
    
    def showEpochEndResult(self,epoch,result):
        print("Epoch[{}] avg_loss : {:.4f} avg_acc : {:.4f}".format(epoch,result['avg_loss'],result['avg_acc']))


def accuracy(outputs:Tensor,labels:[int]):
    _,predIndex = torch.max(outputs,dim=1)
    # How Many Labels predicted correctly / length of total items
    return torch.tensor(torch.sum(predIndex==labels).item()/len(predIndex))

model = MnistModel(input_size,output_size,F.cross_entropy,accuracy)

def loadModel(model:MnistModel):
    if os.path.exists("./store/model.pth"):
        model.load_state_dict(torch.load("./store/model.pth"))

def saveModel(model:MnistModel):
    if(not os.path.exists("./store")):
        os.mkdir("./store")
    torch.save(model.state_dict(),"./store/model.pth")

def evaluateModel(model,dataLoader):
    outputs = [model.validation_step(batch) for batch in dataLoader]
    return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_dl,validation_dl,opt_fn = torch.optim.SGD):
    loadModel(model)
    history = []
    optimizer = opt_fn(model.parameters(),lr)
    for epoch in range(epochs):
        # Training for this <epoch>
        for batch in train_dl:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validating for this <epoch>
        result = evaluateModel(model,val_dl)
        model.showEpochEndResult(epoch,result)
        history.append(result)
    saveModel(model)
    return history


result = fit(5,0.001,model,train_dl,val_dl)

