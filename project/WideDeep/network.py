import torch
import  torch.nn as nn


class Wide(nn.Module):
    def __init__(self,input_dim):
        super(Wide, self).__init__()
        # 全连接层
        self.linear = nn.Linear(in_features=input_dim,out_features=1)

    def forward(self,x):
        x = self.linear(x)
        return x

class Deep(nn.Module):
    def __init__(self,input_dim):
        super(Deep, self).__init__()
         # 构建隐藏层
        self.dnn = nn.ModuleList([
            nn.Linear(input_dim,256),
            nn.Linear(256,128),
            nn.Linear(128,64)
        ])
        self.relu = nn.ReLU()

    def forward(self,x):
        for layer in self.dnn:
            x = layer(x)
            x = self.relu(x)

        return x


class WideDeep(nn.Module):
    def __init__(self,dense_feature_col,sparse_feature_col):
        super(WideDeep, self).__init__()
        # 拆出 稠密特征的长度 便于切割
        self._dense_column_num = dense_feature_col.__len__()
        self._sparse_column_length = sparse_feature_col.__len__()

        # 稀疏特征
        self.sparse_feature = sparse_feature_col

        # Embedding层 这里要 统计每个类别特征的 类别数 作为 nn.Embedding的输入
        self.embedding_layer = nn.ModuleList([
            nn.Embedding(10,8),
            nn.Embedding(42,8)
        ])

    # wide部分
        self._wide = Wide(self._dense_column_num)
    # Deep部分
        self._deep = Deep(self._sparse_column_length)

        # final_linear
        self._final_linear = nn.Linear(64,1)

    def forward(self,x):
        dense_input,sparse_input = (x[:,:self._dense_column_num],x[:self._dense_column_num:])















# model_wide  = Wide(128)
# model_deep = Deep(221)
#
# input_fea = torch.randn(32,221)
# print(model_deep(input_fea).shape)



