import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# 确保路径正确
train_features_path = 'data/processed/train_features.npy'
train_labels_path = 'data/processed/train_labels.npy'
test_features_path = 'data/processed/test_features.npy'
test_ids_path = 'data/processed/test_ids.npy'
output_path = 'outputs/submission_manual.csv'

# 加载数据
X_train = np.load(train_features_path)
y_train = np.load(train_labels_path)
X_test = np.load(test_features_path)
test_ids = np.load(test_ids_path)

# 使用最佳参数训练模型
best_params = {
    'colsample_bytree': 0.6640884574043,
    'learning_rate': 0.0144153788219,
    'max_depth': 12,
    'min_child_samples': 80,
    'min_split_gain': 0.1935627406459,
    'n_estimators': 573,
    'num_leaves': 30,
    'reg_alpha': 1.4785435300368,
    'reg_lambda': 4.3424531986372,
    'subsample': 0.6649544374896
}

model = LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# 预测测试集
test_predictions = model.predict_proba(X_test)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'target_0': test_predictions[:, 0],
    'target_1': test_predictions[:, 1],
    'target_2': test_predictions[:, 2],
    'target_3': test_predictions[:, 3],
    'target_4': test_predictions[:, 4],
    'target_5': test_predictions[:, 5],
    'target_6': test_predictions[:, 6]
})

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 保存提交文件
submission.to_csv(output_path, index=False)
print(f"提交文件已保存到: {output_path}")