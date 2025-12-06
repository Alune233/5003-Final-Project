# 🚀 超级快速开始指南

## 1️⃣ 安装依赖（5 分钟）

```bash
# 核心依赖（所有人都需要）
pip install numpy pandas scikit-learn lightgbm torch transformers matplotlib tqdm

# C同学（TPE）- 需要安装
pip install optuna

# D同学（SMAC/OpenBox）- 需要安装
pip install openbox
# 注意: 在Windows上使用OpenBox，不要安装smac（会失败）
```

## 2️⃣ A 同学：提取并标准化特征（仅运行一次）

```bash
# 步骤1: 提取DeBERTa特征
python main.py --mode extract

# 步骤2: 标准化特征（解决SVM/MLP收敛问题）
python src/preprocess_features.py
```

**生成的文件**: `data/processed/` 目录下的 `.npy` 文件

**完成后**: 分享所有 `.npy` 文件给全组！

## 3️⃣ 运行你的实验（超级简单！）

所有人使用**完全相同**的命令格式：

```bash
python main.py --model [模型] --algo [算法] --n_trials [次数]
```

### A 同学 - Random Search

```bash
python main.py --model lightgbm --algo random --n_trials 50
python main.py --model svm --algo random --n_trials 50
python main.py --model mlp --algo random --n_trials 50
```

### B 同学 - Grid Search

```bash
python main.py --model lightgbm --algo grid
python main.py --model svm --algo grid
python main.py --model mlp --algo grid
```

### C 同学 - TPE (Optuna)

```bash
python main.py --model lightgbm --algo tpe --n_trials 50
python main.py --model svm --algo tpe --n_trials 50
python main.py --model mlp --algo tpe --n_trials 50
```

### D 同学 - SMAC (OpenBox)

```bash
python main.py --model lightgbm --algo smac --n_trials 50
python main.py --model svm --algo smac --n_trials 50
python main.py --model mlp --algo smac --n_trials 50
```

## 4️⃣ 输出文件

运行后会生成：

```
models/
├── {model}_{algo}_history.json    # 优化历史（供D同学收集）
├── {model}_{algo}_history.png     # 收敛曲线
└── {model}_fold_*.pkl              # 训练好的模型

outputs/
└── {model}_{algo}_submission.csv   # Kaggle提交文件
```

## 5️⃣ 提取前 N 轮结果（比较不同轮次）

如果你跑了 50 轮实验，想比较 10 轮、20 轮、50 轮的效果差异：

```bash
# 自动从所有历史JSON文件中提取前10轮
python src/extract_n_trials.py --n_trials 10

# 提取前20轮
python src/extract_n_trials.py --n_trials 20
```

**自动处理**：

-   ✅ 自动扫描 `models/` 目录下所有 `*_history.json` 文件
-   ✅ 提取每个文件的前 N 轮数据
-   ✅ 生成对应的 JSON、PNG 和 submission 文件

**生成的文件**（以 10 轮为例）：

```
models/
├── lightgbm_random_10trials_history.json   # 前10轮历史
├── lightgbm_random_10trials_history.png    # 前10轮收敛曲线
└── ...

outputs/
├── lightgbm_random_10trials_submission.csv # 前10轮最佳结果
└── ...
```

**用途**：无需重新跑实验，直接从 50 轮的结果中提取 10 轮、20 轮的数据进行对比分析！

## 6️⃣ 调整搜索空间（可选）

如果需要修改参数范围，编辑 `config/search_spaces.json`：

```json
{
    "lightgbm": {
        "num_leaves": {
            "type": "int",
            "low": 20, // 修改这里的最小值
            "high": 150, // 修改这里的最大值
            "log": false
        },
        "learning_rate": {
            "type": "float",
            "low": 0.01, // 修改这里
            "high": 0.3, // 修改这里
            "log": true // true表示对数尺度采样
        }
        // ... 其他参数
    }
}
```

**说明**：

-   `type: "int"` - 整数参数
-   `type: "float"` - 浮点数参数
-   `type: "categorical"` - 分类参数（从 choices 中选择）
-   `log: true` - 对数尺度采样（适用于学习率等参数）
-   `log: false` - 线性尺度采样

修改后直接运行实验即可，无需重启程序。

**B 同学注意**：Grid Search 的网格在 `src/hpo/grid_search.py` 的 `_create_param_grid()` 方法中定义。如需调整，编辑该文件第 60-90 行左右的参数网格。

## 7️⃣ D 同学：收集结果

收集所有队友的 `*_history.json` 文件，然后：

1. 创建性能对比表（4×3 矩阵）
2. 绘制收敛曲线（每个模型一张图，4 条线）
3. 分析时间 vs 性能

## 8️⃣ 时间安排

-   **12/03 今天**: 全员完成各自的 3 个模型实验
-   **12/04 明天**: D 同学收集数据，开始分析
-   **12/05**: 完成报告初稿
-   **12/06**: 整合报告
-   **12/07**: 提交

## 9️⃣ 常见问题

### Q: 遇到错误怎么办？

```bash
# 如果提示缺少optuna
pip install optuna

# 如果提示缺少openbox
pip install openbox

# 如果遇到其他依赖问题
pip install -r requirements.txt
```

### Q: 实验要跑多久？

-   LightGBM: 每个 trial 约 1-2 分钟，50 个 trial 约 1-2 小时
-   SVM: 每个 trial 约 2-5 分钟，较慢
-   MLP: 每个 trial 约 3-5 分钟，较慢

### Q: 可以调整试验次数吗？

可以！如果时间不够：

```bash
# 减少到30次试验
python main.py --model lightgbm --algo random --n_trials 30
```

### Q: Grid Search 需要设置 n_trials 吗？

不需要！Grid Search 会自动遍历所有组合：

```bash
# 直接运行，无需指定n_trials
python main.py --model lightgbm --algo grid
```

## 🎉 就这么简单！

**你不需要**：

-   ❌ 写任何代码
-   ❌ 了解交叉验证细节
-   ❌ 手动保存结果
-   ❌ 配置复杂参数

**你只需要**：

-   ✅ 安装依赖
-   ✅ 运行一行命令（3 个参数）
-   ✅ 等待结果

**超级精简！无脑运行！** 🚀
