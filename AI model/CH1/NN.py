
#模擬神經元
import math
# 定義神經元的輸入權重和閾值
weights = [0.5, 0.3, -0.2]  # 輸入權重

# 簡單的激活函數
threshold = 0.4  # 閾值

# 定義神經元的輸入
inputs = [0.7, 0.2, 0.5]  # 輸入值

# 計算神經元的加權和
weighted_sum = 0
for i in range(len(weights)):
    weighted_sum += weights[i] * inputs[i]

# 判斷神經元是否激活（使用閾值函數）
if weighted_sum >= threshold:
    output = 1  # 激活
else:
    output = 0  # 不激活

# 印出神經元的輸出
print("神經元的輸出:", output)
