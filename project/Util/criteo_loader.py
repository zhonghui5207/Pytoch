import pandas as pd
import numpy as np


def DataProcess():
    train_file = "../Data/criteo/origin_data/train.csv"
    test_file = "../Data/criteo/origin_data/test.csv"
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    print(train_data.shape,test_data.shape)
    #  保留 训练集的label
    label = train_data['label']
    del train_data['label']
    # 合并 train 和test 一起做 数据预处理
    data_df = pd.concat((train_data,test_data))

DataProcess()