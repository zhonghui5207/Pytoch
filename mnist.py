import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

)

train_data = datasets.MNIST(
    root="../datasets/mnist/",
    train=True,
    download=False,
    transform=transform
)

test_data = datasets.MNIST(
    root="../datasets/mnist/",
    train=False,
    download=False,
    transform=transform
)
train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True)
test_loader = DataLoader(dataset=test_data,
                          batch_size=64,
                          shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(784,512)
        self.l2 = nn.Linear(512,256)
        self.l3 = nn.Linear(256,128)
        self.l4 = nn.Linear(128,64)
        self.l5 = nn.Linear(64,10)
        self.activate = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,784)
        x = self.activate(self.l1(x))
        x = self.activate(self.l2(x))
        x = self.activate(self.l3(x))
        x = self.activate(self.l4(x))
        return self.l5(x)

model = Model()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),0.01,momentum=0.5)

def train(epoch):
    run_loss = 0
    for batch_id,data in enumerate(train_loader,0):
        data_x,data_y = data
        pred_y = model(data_x)
        loss = criterion(pred_y,data_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()

        if batch_id % 300 == 299:
            print("[%d,%5d],loss:%.2f" %(epoch,batch_id,run_loss/300))

            run_loss =0
def test():
    collect =0
    total = 0

    for _,data in enumerate(test_loader,0):
        with torch.no_grad():
            data_x,data_y = data
            pred_y = model(data_x)

            _,predicted = torch.max(pred_y,dim=-1)
            total += data_y.size(0)
            collect += (data_y==predicted).sum().item()

    print("acc:%.2f%%",100 * collect/total)








if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()



