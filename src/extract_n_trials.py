"""
从完整的HPO历史中提取前N轮的结果

自动扫描models目录下所有的历史JSON文件并提取前N轮

使用方法:
    python src/extract_n_trials.py --n_trials 10
    python src/extract_n_trials.py --n_trials 20
"""

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt
import logging

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_history(history, save_path):
    """绘制优化历史"""
    trials = [h['trial'] for h in history]
    scores = [h['score'] for h in history]
    
    # 计算累积最佳得分
    best_scores = []
    current_best = float('inf')
    for score in scores:
        current_best = min(current_best, score)
        best_scores.append(current_best)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 子图1: 所有试验的得分
    plt.subplot(1, 2, 1)
    plt.scatter(trials, scores, alpha=0.6, s=30)
    plt.plot(trials, best_scores, 'r-', linewidth=2, label='Best Score')
    plt.xlabel('Trial')
    plt.ylabel('Score (logloss)')
    plt.title(f'Optimization History (First {len(history)} Trials)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 得分分布
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=min(30, len(scores)), alpha=0.7, edgecolor='black')
    plt.axvline(min(scores), color='r', linestyle='--', linewidth=2, 
                label=f'Best: {min(scores):.6f}')
    plt.xlabel('Score (logloss)')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 优化历史图已保存: {save_path}")


def process_json_file(json_path, n_trials):
    """处理单个JSON文件"""
    # 解析文件名以获取模型和算法信息
    filename = os.path.basename(json_path)
    parts = filename.replace('_history.json', '').split('_')
    if len(parts) >= 2:
        model_name = parts[0]
        algo_name = parts[1]
    else:
        logger.error(f"无法从文件名解析模型和算法信息: {filename}")
        return False
    
    logger.info("\n" + "="*80)
    logger.info(f"处理: {filename}")
    logger.info(f"模型: {model_name} | 算法: {algo_name} | 提取前{n_trials}轮")
    logger.info("="*80)
    
    # 加载历史数据
    logger.info(f"加载历史数据...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            full_history = json.load(f)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return False
    
    logger.info(f"完整历史包含 {len(full_history)} 轮试验")
    
    if n_trials > len(full_history):
        logger.warning(f"请求的轮次({n_trials})超过历史记录({len(full_history)})，跳过")
        return False
    
    # 提取前N轮
    n_history = full_history[:n_trials]
    logger.info(f"提取前 {len(n_history)} 轮")
    
    # 找到最佳参数
    best_idx = min(range(len(n_history)), key=lambda i: n_history[i]['score'])
    best_params = n_history[best_idx]['params']
    best_score = n_history[best_idx]['score']
    
    logger.info(f"前{n_trials}轮最佳得分: {best_score:.6f} (Trial {best_idx})")
    
    # 保存前N轮的历史JSON
    output_dir = os.path.dirname(json_path)
    n_json_path = os.path.join(
        output_dir,
        f'{model_name}_{algo_name}_{n_trials}trials_history.json'
    )
    with open(n_json_path, 'w', encoding='utf-8') as f:
        json.dump(n_history, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 历史JSON已保存: {n_json_path}")
    
    # 绘制优化历史图
    n_plot_path = os.path.join(
        output_dir,
        f'{model_name}_{algo_name}_{n_trials}trials_history.png'
    )
    plot_history(n_history, n_plot_path)
    
    # 用最佳参数训练模型并生成submission（无需交叉验证）
    logger.info("使用最佳参数训练模型并预测...")
    
    # 加载数据
    train_feature_path = 'data/processed/train_features.npy'
    test_feature_path = 'data/processed/test_features.npy'
    train_labels_path = 'data/processed/train_labels.npy'
    test_ids_path = 'data/processed/test_ids.npy'
    
    if not os.path.exists(train_feature_path):
        logger.error("特征文件不存在，请先运行特征提取")
        return False
    
    try:
        X_train = np.load(train_feature_path)
        y_train = np.load(train_labels_path)
        X_test = np.load(test_feature_path)
        test_ids = np.load(test_ids_path)
        
        # 获取模型类
        ModelClass = get_model(model_name)
        
        # 直接用全部训练数据训练一个模型（无需交叉验证）
        model = ModelClass(params=best_params, n_folds=1)  # n_folds=1表示不做CV
        
        # 对于MLP，需要手动处理标准化
        if model_name == 'mlp':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 创建并训练模型
            single_model = model._create_model(best_params)
            single_model.fit(X_train_scaled, y_train)
            test_predictions = single_model.predict_proba(X_test_scaled)
        else:
            # 对于LightGBM和SVM，直接训练
            single_model = model._create_model(best_params)
            if model_name == 'lightgbm':
                import lightgbm as lgb
                train_data = lgb.Dataset(X_train, label=y_train)
                single_model = lgb.train(single_model, train_data, num_boost_round=best_params.get('n_estimators', 500))
                test_predictions = single_model.predict(X_test, num_iteration=single_model.best_iteration if hasattr(single_model, 'best_iteration') else None)
            else:  # SVM
                single_model.fit(X_train, y_train)
                test_predictions = single_model.predict_proba(X_test)
        
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
        
        # 保存submission
        output_path = f'outputs/{model_name}_{algo_name}_{n_trials}trials_submission.csv'
        os.makedirs('outputs', exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        logger.info(f"✓ 历史JSON: {n_json_path}")
        logger.info(f"✓ 优化曲线: {n_plot_path}")
        logger.info(f"✓ 提交文件: {output_path}")
        logger.info(f"✓ 最佳CV得分: {best_score:.6f} (来自历史记录)")
        
        return True
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='从完整HPO历史中提取前N轮结果（自动扫描所有历史文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/extract_n_trials.py --n_trials 10
  python src/extract_n_trials.py --n_trials 20
        """
    )
    
    parser.add_argument('--n_trials', type=int, required=True,
                       help='提取前N轮')
    
    args = parser.parse_args()
    
    # 查找所有历史JSON文件
    json_files = glob.glob('models/*_history.json')
    
    # 过滤掉已经处理过的文件（包含trials的）
    json_files = [f for f in json_files if 'trials' not in f]
    
    if not json_files:
        logger.error("未找到任何历史JSON文件")
        logger.error("请确保已运行过HPO实验，并且models目录下有*_history.json文件")
        return 1
    
    logger.info("="*80)
    logger.info(f"自动提取前{args.n_trials}轮结果")
    logger.info("="*80)
    logger.info(f"\n找到 {len(json_files)} 个历史文件:")
    for f in json_files:
        logger.info(f"  - {f}")
    
    # 处理每个文件
    success_count = 0
    for json_file in json_files:
        if process_json_file(json_file, args.n_trials):
            success_count += 1
    
    logger.info("\n" + "="*80)
    logger.info(f"完成! 成功处理 {success_count}/{len(json_files)} 个文件")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

