import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets as dataset
from torch.utils.data import DataLoader

transform = transform.Compose(
    [
        transform.ToTensor(),
        transform.Normalize((0.1307,),(0.3086))
    ]
)

train_data = dataset.MNIST(
    root='../datasets/mnist/',
    train=True,
    download=False,
    transform=transform
)
test_data = dataset.MNIST(
    root='../datasets/mnist/',
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=64
)

test_loader = DataLoader(
    dataset=test_data,
    shuffle=True,
    batch_size=64
)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,10,(5,5))
        self.conv2 = nn.Conv2d(10,20,(5,5))
        self.max_1 = nn.MaxPool2d(2)
        self.activate = nn.ReLU()
        self.linear = nn.Linear(320,10)

    def forward(self,x):
        batchsize = x.size(0)
        #print("batch-size",batchsize)
        x = self.activate(self.max_1(self.conv1(x)))
        #print("x1",x.shape)
        x = self.activate(self.max_1(self.conv2(x)))
        #print("x2",x.shape)
        x = x.view(batchsize,-1)
        #print("x3",x.shape)
        x = self.linear(x)
        return x

model = Model()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),0.05,momentum=0.5)

def train(epoch):
    run_loss = 0
    for batch_id,data in enumerate(train_loader,0):
        inputs,target = data
        pred_y = model(inputs)
        loss = criterion(pred_y,target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        run_loss += loss.item()

        if batch_id % 300 == 299:
            print("[%d,%5d],loss：%.2f" %(epoch+1,batch_id+1,run_loss/300))
            run_loss =0

def test():
    correct =0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input,target = data

            pred_y = model(input)
            _,predicted = torch.max(pred_y,dim=-1)
            total += target.size(0)
            correct += (target == predicted).sum().item()

    print("acc: %.2f%%,[%d,%d]"%(100 * correct / total,correct,total))
if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()