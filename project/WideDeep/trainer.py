import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
from project.WideDeep import network


class Trainer(object):
    def __init__(self, model):
        self._model = model
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=0.1)
        self._loss_func = torch.nn.BCELoss()

    def _train_single_batch(self, x, label):
        """
        对一个小批次进行训练
        :param x:
        :param label:
        :return:
        """
        self._optimizer.zero_grad()

        pre_y = self._model(x)

        loss = self._loss_func(pre_y.view(-1), label)

        loss.backward()

        self._optimizer.step()

        loss = loss.item()

        return loss, pre_y

    def _train_an_epoch(self, train_loader, epoch_id):
        """
        训练一个epoch  把所有的样本都训练一遍
        :param train_loader:
        :param epoch_id:
        :return:
        """
        # 设置模型为训练模式，启用 dropout 以及 batch normalization 归一化
        self._model.train()
        total = 0
        for batch_id, (x, label) in enumerate(train_loader):
            x = Variable(x)
            label = Variable(label)
            loss, pre_y = self._train_single_batch(x, label)

            total += loss
            print("1:", epoch_id, batch_id, loss)
        print("2:", epoch_id, total)

    def train(self, train_dataset):
        for epoch in range(3):
            dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
            self._train_an_epoch(dataloader, epoch)


input_wide = torch.randn(32000,221)
input_deep_num1 = torch.randint(0,10,(32000,1))
input_deep_num2 = torch.randint(0,42,(32000,1))
input_deep = torch.cat((input_deep_num1,input_deep_num2),dim=-1)
input = torch.cat([input_wide,input_deep],axis=-1)
label = torch.randint(0,2,(32000,1)).float()
label = label.view(-1)

train_dataset = TensorDataset(input,label)
widedeep = network.WideDeep(221,237)
trainer = Trainer(model=widedeep)
trainer.train(train_dataset)