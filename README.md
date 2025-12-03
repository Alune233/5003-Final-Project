# LLM 分类器

> **核心理念**：特征提取 + baseline + 不同HPO，**你**只需关注 **HPO的工作（见HPO部分）**

---

## 说明

### 已完成

- ✅ DeBERTa 特征提取
- ✅ LightGBM 训练框架
- ✅ Random Search baseline

## 开始

### 环境安装

```bash
# 1. 安装
conda create -n llm_classifier python=3.9

conda activate llm_classifier

pip install -r requirements.txt
```

### HPO

实现自己的 HPO 算法，3 步搞定：

#### 1. 创建算法文件 `src/hpo/your_algo.py`

```python
from .base_hpo import BaseHPO
import numpy as np

class YourAlgorithm(BaseHPO):
    def __init__(self, n_trials=50, cv_folds=5, random_state=42):
        super().__init__(n_trials, cv_folds, random_state)

        # 在算法内部定义搜索空间（每个算法自己定义！）
        # 不同算法可以用不同的方式定义
        self.search_space = {
            'num_leaves': ('int', 20, 150),
            'learning_rate': ('float_log', 0.01, 0.3),
            # ... 你的其他参数
        }

    def suggest_params(self):
        # 建议参数
        params = {}
        # ... 你的实现
        params.update(self.fixed_params)
        return params

    def optimize(self, objective_function, verbose=True):
        # 执行优化
        for i in range(self.n_trials):
            params = self.suggest_params()
            score = objective_function(params)
            self.history.append({'trial': i, 'params': params, 'score': score})
        return self.get_best_params(), self.get_best_score()
```

#### 2. 注册算法到 `src/hpo/__init__.py`

```python
from .grid_search import GridSearch
from .your_algo import YourAlgorithm

AVAILABLE_ALGORITHMS = {
    'random': RandomSearch,
    'grid': GridSearch,
    'your_algo': YourAlgorithm,  # 加这一行
}
```

#### 3. 运行

```bash
python main.py --mode hpo --algo your_algo
```

---

## 命令行参数

**只有 2 个参数，超简单：**

| 参数       | 说明           | 默认值 |
| ---------- | -------------- | ------ |
| `--mode` | extract 或 hpo | hpo    |
| `--algo` | HPO 算法名称   | random |

**仅此而已！** 所有其他配置（搜索次数、CV 折数等）在 自己的HPO文件 中设置。

**示例**：

```bash
# 默认运行（HPO with random算法）
python main.py

# Grid Search
python main.py --algo grid

# 提取特征
python main.py --mode extract

# 使用其他算法
python main.py --algo bayesian
```

**简单直接，不需要额外的配置文件！**

---

## 项目结构

```
.
├── src/
│   ├── feature_extraction.py     # 特征提取（完成）
│   ├── train.py                  # LightGBM训练（完成）
│   └── hpo/
│       ├── __init__.py           # 算法注册表（在这里注册）
│       ├── base_hpo.py           # 基类（继承它）
│       └── random_search.py      # Random Search（参考）
├── data/
│   ├── raw/                      # 原始数据
│   └── processed/                # 提取的特征（共享）
├── main.py                       # 主入口
└── README.md                     # 说明
```

---

## 为什么每个算法自己定义搜索空间？

不同的 HPO 算法需要不同的搜索空间定义方式：

- **Random Search**: 简单的范围 `('int', 20, 150)`
- **Grid Search**: 离散网格 `[20, 50, 100, 150]`
- **TPE (Hyperopt)**: `hp.uniform()`, `hp.loguniform()` 对象
- **SMAC**: `ConfigSpace` 对象
- **OpenBox**: `Space` 对象

统一的 JSON 无法满足所有需求，所以让每个算法在自己代码中灵活定义！

---

## 输出文件

```
outputs/
└── submission.csv              # Kaggle提交文件

models/
├── lgb_fold_*.txt             # 训练好的模型
└── {algorithm}_history.json   # HPO历史记录
```

---

## 常见问题

**Q: 找不到特征文件？**

```bash
data/processed 文件夹下
```

**Q: 怎么查看可用算法？**

```bash
python main.py --algo unknown  # 会提示所有可用算法
```

---

## 实验记录建议

| 实验 ID | 算法     | trials | CV Score | 备注     |
| ------- | -------- | ------ | -------- | -------- |
| exp01   | random   | 50     | 0.XXXX   | baseline |
| exp02   | bayesian | 50     | 0.XXXX   | ...      |
| exp03   | tpe      | 50     | 0.XXXX   | ...      |
