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
    def __init__(self,num_dense_feature,num_feature):
        super(WideDeep, self).__init__()
        # 拆出 稠密特征的长度 便于切割
        self._dense_column_num = num_dense_feature
        #self._dense_column_num = dense_feature_col.__len__()
        self._sparse_column_length = num_feature

        # 稀疏特征
        #self.sparse_feature = sparse_feature_col

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
        # 分割出 稠密特征和稀疏特征
        dense_input,sparse_input = (x[:,:self._dense_column_num],x[:,self._dense_column_num:])

        sparse_input = sparse_input.long()
        sparse_embeds = [self.embedding_layer[i](sparse_input[:,i]) for i in range(2)]
        # 合并 sparse embedds
        sparse_embeds = torch.cat(sparse_embeds,axis=-1)
        #print("sparse_embeds",sparse_embeds.shape,sparse_embeds)

        # 合并 稠密和稀疏特征
        deep_in = torch.cat((sparse_embeds,dense_input),dim=-1)
        #   wide 的输出
        wide_out = self._wide(dense_input)
        #  deep 的输出  （32,64）
        deep_out = self._deep(deep_in)
        #  经过最后一层
        deep_out = self._final_linear(deep_out)
        # 一定要保持 wide 和 deep 的形状一直 才能 后面做平均
        assert (wide_out.shape == deep_out.shape)

        out_puts = torch.sigmoid(0.5 * (wide_out + deep_out))
        return out_puts


# model_wide  = Wide(128)
# model_deep = Deep(221)
#
# input_fea = torch.randn(32,221)
# print(model_deep(input_fea).shape)

model = WideDeep(221,237)
input_wide = torch.randn(32,221)
input_deep_num1 = torch.randint(0,10,(32,1))
input_deep_num2 = torch.randint(0,42,(32,1))
input_deep = torch.cat((input_deep_num1,input_deep_num2),dim=-1)
input = torch.cat([input_wide,input_deep],axis=-1)
pre = model(input)
# print(pre)



