# 01 背包问题
def weightbag(weight, value, bagweight):
    # 定义dp数组
    dp = [[0 for _ in range(bagweight+1)] for _ in range(len(weight))]
    print(dp)
    # 数组初始化
    for j in range(1,bagweight+1):
        if weight[0] <= j:
            dp[0][j] = value[0]
    # 遍历顺序  先遍历物品在遍历背包
    for i in range(1,len(weight)):
        for j in range(1,bagweight+1):
            # 如果当前物品重量大于背包重量 则不加改物品 当前价值就是最大价值
            if weight[i] > j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]] + value[i])
    return dp[-1][-1]




if __name__ == '__main__':
    weight = [1, 3, 4,5]
    value = [15, 20, 30,40]
    bagweight = 5
    print(weightbag(weight, value, bagweight))
    # dp = [[0 for _ in range(4)]] * 3
    # print(dp)
    
