import argparse
import numpy as np
import logging
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import extract_and_save_features
from train import objective_function_factory
from hpo import get_hpo_algorithm, list_algorithms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_feature_extraction(args):
    """运行特征提取"""
    logger.info("="*80)
    logger.info("DeBERTa特征提取")
    logger.info("="*80)
    
    extract_and_save_features(
        train_path=args.train_path,
        test_path=args.test_path,
        train_feature_path=args.train_feature_path,
        test_feature_path=args.test_feature_path,
        train_labels_path=args.train_labels_path,
        test_ids_path=args.test_ids_path,
        model_name=args.deberta_model,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    logger.info("\n特征提取完成！")


def run_hpo(args):
    """
    运行超参数优化（核心函数）
    自动选择并运行指定的HPO算法
    所有优化参数从配置文件读取
    """
    logger.info("="*80)
    logger.info(f"HPO超参数优化 - 算法: {args.hpo_algorithm}")
    logger.info("="*80)
    
    # 检查特征文件是否存在
    if not os.path.exists(args.train_feature_path):
        logger.error(f"特征文件不存在: {args.train_feature_path}")
        logger.error("请先运行: python main.py --mode extract")
        return None, None
    
    # 加载训练数据
    logger.info("加载训练特征...")
    X_train = np.load(args.train_feature_path)
    y_train = np.load(args.train_labels_path)
    X_test = np.load(args.test_feature_path)
    test_ids = np.load(args.test_ids_path)
    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 自动获取HPO算法类（根据命令行参数）
    HPOClass = get_hpo_algorithm(args.hpo_algorithm)

    # 初始化HPO算法（参数由各HPO算法自己设置）
    optimizer_kwargs = {}
    if getattr(args, 'use_history', False):
        optimizer_kwargs['use_history'] = True
    if getattr(args, 'history_path', None):
        optimizer_kwargs['initial_history_path'] = args.history_path
    optimizer = HPOClass(**optimizer_kwargs)
    
    # 创建目标函数（使用HPO算法自己设置的cv_folds）
    objective_fn = objective_function_factory(
        X_train, 
        y_train,
        n_folds=optimizer.cv_folds,
        early_stopping_rounds=50
    )
    
    # 执行优化（参数从配置文件读取）
    best_params, best_score = optimizer.optimize(
        objective_function=objective_fn,
        verbose=True
    )
    
    # 保存结果
    os.makedirs(args.model_dir, exist_ok=True)
    history_path = os.path.join(args.model_dir, f'{args.hpo_algorithm}_history.json')
    optimizer.save_history(history_path)
    logger.info(f"优化历史已保存: {history_path}")

    # 额外再保存一份到 outputs 目录，便于后续上传/查看
    outputs_dir = os.path.dirname(args.output_path) or 'outputs'
    os.makedirs(outputs_dir, exist_ok=True)
    outputs_history_path = os.path.join(outputs_dir, f'{args.hpo_algorithm}_history.json')
    optimizer.save_history(outputs_history_path)
    logger.info(f"优化历史也已保存到: {outputs_history_path}")
    
    # 绘制优化历史（如果支持）
    try:
        plot_path = os.path.join(args.model_dir, f'{args.hpo_algorithm}_history.png')
        optimizer.plot_optimization_history(plot_path)
        logger.info(f"可视化图已保存: {plot_path}")
    except Exception as e:
        logger.debug(f"无法绘制优化历史: {e}")
    
    # 使用最佳参数训练最终模型并生成提交文件
    logger.info("\n" + "="*80)
    logger.info("使用最佳参数训练最终模型")
    logger.info("="*80)
    
    from train import LightGBMTrainer
    import pandas as pd
    
    trainer = LightGBMTrainer(params=best_params, n_folds=optimizer.cv_folds)
    final_cv_score = trainer.train_with_cv(X_train, y_train)
    
    # 保存模型
    trainer.save_models(args.model_dir)
    
    # 预测测试集
    logger.info("\n预测测试集...")
    test_predictions = trainer.predict(X_test)
    
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
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存提交文件
    submission.to_csv(args.output_path, index=False)
    
    logger.info(f"\n提交文件已保存: {args.output_path}")
    logger.info(f"最终CV得分: {final_cv_score:.6f}")
    
    # 额外保存一份带时间戳的副本，防止覆盖
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(output_dir, f"submission_{args.hpo_algorithm}_{timestamp}.csv")
    submission.to_csv(backup_path, index=False)
    logger.info(f"提交文件副本已保存: {backup_path}")
    
    return best_params, best_score


def main():
    """主函数 - 极简版，只有2个参数"""
    parser = argparse.ArgumentParser(
        description='LLM分类器 - 极简版Baseline（只有2个参数！）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                            # 默认运行HPO（random算法）
  python main.py --mode extract             # 提取特征
  python main.py --algo bayesian            # 使用其他算法
        """
    )
    
    # ===== 只保留2个核心参数 =====
    parser.add_argument('--mode', type=str, default='hpo',
                       choices=['extract', 'hpo'],
                       help='运行模式: extract(特征提取) 或 hpo(超参数优化)')
    
    parser.add_argument('--algo', type=str, default='random',
                       help=f'HPO算法名称，可用: {list_algorithms()}')
    parser.add_argument('--use_history', action='store_true', help='是否使用历史记录来缩小搜索空间（适用于SMAC3）')
    parser.add_argument('--history_path', type=str, default=None, help='历史文件路径（用于warm start）')
    
    args = parser.parse_args()
    
    # ===== 固定配置（直接在代码中定义，简单直接）=====
    args.hpo_algorithm = args.algo  # 统一使用algo
    args.train_path = 'data/raw/train.csv'
    args.test_path = 'data/raw/test.csv'
    args.train_feature_path = 'data/processed/train_features.npy'
    args.test_feature_path = 'data/processed/test_features.npy'
    args.train_labels_path = 'data/processed/train_labels.npy'
    args.test_ids_path = 'data/processed/test_ids.npy'
    args.output_path = 'outputs/submission.csv'
    args.model_dir = 'models'
    args.deberta_model = 'microsoft/deberta-v3-base'
    args.max_length = 512
    args.batch_size = 8
    
    # 打印配置
    logger.info("="*80)
    logger.info("LLM分类器 - 极简Baseline")
    logger.info("="*80)
    logger.info(f"运行模式: {args.mode}")
    if args.mode == 'hpo':
        logger.info(f"HPO算法: {args.hpo_algorithm}")
    logger.info("="*80 + "\n")
    
    try:
        if args.mode == 'extract':
            # 特征提取
            run_feature_extraction(args)
            
        elif args.mode == 'hpo':
            # HPO优化
            run_hpo(args)
        
        logger.info("\n" + "="*80)
        logger.info("任务完成!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n发生错误: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

