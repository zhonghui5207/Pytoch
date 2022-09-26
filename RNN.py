"""
RNNcell 与 RNN 结构上的区别和 训练形式上的区别

"""
import torch.nn as nn
import torch

# 定义参数
batch_size = 1
input_size = 4
hidden_size = 4
num_layer = 1
seq_len = 4

idx_char = ['e','h','l','o']
data_x = [1,0,2,2,3]
data_y = [3,1,2,3,2]

one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]
                  ]

x_one_hot = [one_hot_lookup[i] for i in data_x]
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels = torch.LongTensor(data_y).view(-1)
print("input.shape,input,",(inputs.shape,inputs))
class Model(nn.Module):
    def __init__(self,batch_size,input_size,hidden_size,num_layer):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.rnn = nn.RNN(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layer)

    def forward(self,inputs):
        hx = torch.zeros(self.num_layer,self.batch_size,self.hidden_size)
        out,_  = self.rnn(inputs,hx)
        return out.view(-1,self.hidden_size)

net = Model(batch_size,input_size,hidden_size,num_layer)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(10):

    optimizer.zero_grad()

    target = net(inputs)

    loss = criterion(target,labels)

    loss.backward()

    optimizer.step()
    #print(target.shape,target)
    _,idx = target.max(dim=1)

    print("Prodicted: ",''.join([idx_char[i] for i in idx]),end='')
    print(', epoch[%d/10],loss:%.2f'%(epoch+1,loss.item()))
