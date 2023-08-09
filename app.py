import torch
import numpy
import pandas
from torch import Tensor
from torch import tensor
import torchvision
import torchvision.transforms as visionTransform
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import time
TensorToImageTransform=visionTransform.ToPILImage(mode="L")
dataset = MNIST('./data',download=False,transform=visionTransform.ToTensor())
test_dataset = MNIST('./data',train=False,transform=visionTransform.ToTensor())
batch_size=1024
train_ds,val_ds = random_split(dataset=dataset,lengths=[50000,10000])
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)
test_dl = DataLoader(test_dataset)

input_size=28*28
hidden_size = 32
output_size=10

def getDefaultDevice():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    return torch.device('cpu')

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device) -> None:
        self.dl=dl
        self.device=device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)

    def __len__(self):
        return len(self.dl)

device = getDefaultDevice()
train_dl = DeviceDataLoader(train_dl,device)
val_dl = DeviceDataLoader(val_dl,device)
test_dl = DeviceDataLoader(test_dl,device)

class MnistModel(nn.Module):
    def __init__(self, input_size:int,hidden_size:int,output_size:int,lossFn,accuracyFn) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
        self.lossFn =lossFn
        self.accuracyFn = accuracyFn
    
    def forward(self,inputBatch:Tensor):
        print(inputBatch.shape)
        inputBatch=inputBatch.view(inputBatch.size(0),-1)
        out = self.linear(inputBatch)
        out = F.relu(out)
        out = self.linear2(out)
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
            'val_loss':loss.detach(),
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

model = MnistModel(input_size,hidden_size,output_size,F.cross_entropy,accuracy)

def loadModel(model:MnistModel):
    if os.path.exists("./store/model.pth"):
        model.load_state_dict(torch.load("./store/model.pth"))
    to_device(model,device)

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

startTime = time.time()
result = fit(0,0.5,model,train_dl,val_dl)
print("Total Time = {}".format(time.time()-startTime))

def testRandom():
    random.seed(time.time()*1000)
    test_ds_len = len(test_dataset)
    random_test_index = random.randint(0,test_ds_len)
    image,label = test_dataset[random_test_index]
    testInput = image.unsqueeze(0)
    preds = model(testInput)
    print(torch.argmax(preds))
    print("Should Predict {}".format(label))
    plt.imshow(TensorToImageTransform(image))
    plt.show()

# testRandom()

def testAll():
    correct=tensor([0],device=device)
    for images,lable in test_dl:
        preds = model(images.unsqueeze(0))
        correct+=torch.argmax(preds).item()==lable
    print("Test Score = {:.2f}%".format((correct.detach().item()/len(test_dl))*100))

# testRandom()
testAll()