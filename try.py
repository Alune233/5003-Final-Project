import lightgbm as lgb
import numpy as np

# 随机数据
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

data = lgb.Dataset(X, label=y)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device': 'gpu'  # 关键：启用 GPU
}

try:
    model = lgb.train(params, data, num_boost_round=10)
    print("LightGBM GPU 可用，训练成功！")
except Exception as e:
    print("GPU 不可用或训练失败：", e)
