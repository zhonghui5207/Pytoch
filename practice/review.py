# 线性回归实现
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.functional as F

# 准备数据
# batch_size = 10
# num_input = 2
# num_example = 1000
# true_w =[0.2,-3.4]
# true_b = 4.5
# features = torch.tensor(np.random.normal(0,0.1,(num_example,num_input)),dtype=torch.float)
# #print(features)
# labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
# # 增加偏置
# labels += torch.tensor(np.random.normal(0,0.001,size=labels.size()),dtype=torch.float)
# # 读取数据
# datasets = Data.TensorDataset(features,labels)
#
# data_iter = Data.DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
# # 定义模型
# class LinearModel(nn.Module):
#     def __init__(self,feature_n):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(feature_n,1)
#
#     def forward(self,x):
#         y = self.linear(x)
#         return y
# model = LinearModel(feature_n=num_input)
# # 定义损失函数
# criterion = nn.MSELoss()
#
# # 定义优化器
# optimizer = torch.optim.SGD(model.parameters(),lr=0.03)
#
# # 训练模型
# for epoch in range(3):
#     for X,y in data_iter:
#         out_put = model(X)
#         #print(out_put.shape,y.shape)
#         loss = criterion(out_put,y.view(-1,1))
#         # 梯度清零
#         optimizer.zero_grad()
#         # 反向传播
#         loss.backward()
#         # 梯度迭代
#         optimizer.step()
#
#     print("epoch:%d loss:%.2f" %(epoch+1,loss.item()))
# print(true_w,model.parameters())

# softmax 回归模型
m = nn.Softmax(dim=1)
print(m)
input = torch.randn(2, 3)
print(input)
output = m(input)
print(output)

