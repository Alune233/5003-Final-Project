import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """
    LightGBM训练器
    支持交叉验证和超参数优化
    """
    
    def __init__(self, params=None, n_folds=5, random_state=42):
        """
        初始化训练器
        
        Args:
            params: LightGBM参数字典
            n_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.params = params or self._get_default_params()
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []  # 存储每个fold的模型
        self.feature_importance = None
        
        logger.info(f"初始化LightGBM训练器，{n_folds}折交叉验证")
    
    def _get_default_params(self):
        """获取默认参数"""
        return {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
    
    def train_with_cv(self, X, y, early_stopping_rounds=50, verbose=100, suppress_lgb_warnings=True, force_col_wise=False):
        """
        使用交叉验证训练模型
        
        Args:
            X: 特征数组，形状 (n_samples, n_features)
            y: 标签数组，形状 (n_samples,)
            early_stopping_rounds: 早停轮数
            verbose: 日志输出频率
            
        Returns:
            float: 交叉验证平均得分(logloss)
        """
        # 确保参数中包含 num_class
        if 'num_class' not in self.params:
            self.params['num_class'] = 7
        if 'objective' not in self.params:
            self.params['objective'] = 'multiclass'
        if 'metric' not in self.params:
            self.params['metric'] = 'multi_logloss'

        # 如果用户想强制 col-wise，设置到参数里以避免开销检测 Info 输出
        if force_col_wise:
            self.params['force_col_wise'] = True

        logger.info("="*60)
        logger.info("开始交叉验证训练")
        logger.info(f"训练样本数: {len(X)}, 特征维度: {X.shape[1]}")
        logger.info("="*60)
        
        # 初始化
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        oof_predictions = np.zeros((len(X), 7))  # Out-of-fold预测
        self.models = []
        
        # 添加记录最佳超参数的逻辑
        best_params = None
        best_score = float('inf')
        best_fold = None
        
        # 交叉验证
        # WARNING: 某些 LightGBM 内部警告会持续打印（例如：'No further splits with positive gain'），
        # 这类警告通常是因为训练集中某些节点没有可用的分裂（如样本太少或标签单一）。
        # 如果不希望看到这些警告，可以通过 suppress_lgb_warnings=True 抑制（使用 warnings.filterwarnings）。
        import warnings
        filter_context = None
        if suppress_lgb_warnings:
            # 只忽略特定消息，使用 catch_warnings 以避免全局修改
            filter_context = warnings.catch_warnings()
            filter_context.__enter__()
            warnings.filterwarnings('ignore', message='No further splits with positive gain')
            warnings.filterwarnings('ignore', message='Auto-choosing col-wise multi-threading')
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")
            logger.info(f"{'='*60}")
            
            # 划分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 诊断: 检查每个折是否包含至少两个类别（多分类训练需要）
            unique_train_classes = np.unique(y_train)
            if len(unique_train_classes) < 2:
                logger.warning(f"Fold {fold_idx+1} 跳过训练: 训练集只包含单一类别: {unique_train_classes}")
                # 为保持 oof_predictions 的结构，我们为验证集返回占位预测（one-hot majority）
                maj_class = int(unique_train_classes[0])
                val_pred = np.zeros((len(X_val), self.params.get('num_class', 7)))
                val_pred[:, maj_class] = 1.0
                cv_scores.append(log_loss(y_val, val_pred))
                oof_predictions[val_idx] = val_pred
                continue

            # 诊断: 检查 min_child_samples/min_split_gain 与数据规模的关系
            min_child_samples = self.params.get('min_child_samples', None)
            min_split_gain = self.params.get('min_split_gain', None) or self.params.get('min_gain_to_split', None)
            if min_child_samples and min_child_samples > len(X_train) * 0.5:
                logger.warning(f"Fold {fold_idx+1}: min_child_samples({min_child_samples}) 相对于训练样本 ({len(X_train)}) 太大，可能导致无法分裂")
            if min_split_gain and min_split_gain > 1.0:
                logger.warning(f"Fold {fold_idx+1}: min_split_gain({min_split_gain}) 较大，可能导致缺乏正增益分裂")

            logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
            
            # 创建数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 训练模型 - 如果 verbose<=0 则不输出训练日志（将不会加入 log_evaluation 回调）
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
            if verbose and verbose > 0:
                callbacks.append(lgb.log_evaluation(period=verbose))

            # 如果需要抑制 LightGBM 内部 Warning（如 "No further splits with positive gain"），可捕获 stderr
            if suppress_lgb_warnings:
                import os
                try:
                    # 另外设置 Python logger 对 LightGBM 的 level（Python 侧）
                    import logging as _logging
                    _logging.getLogger('lightgbm').setLevel(_logging.ERROR)
                except Exception:
                    pass

                # 将操作系统级别的 stderr 重定向到 /dev/null，捕获 C++ 层的输出
                devnull_fd = os.open(os.devnull, os.O_RDWR)
                old_stderr_fd = os.dup(2)
                try:
                    os.dup2(devnull_fd, 2)
                    model = lgb.train(
                        self.params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        valid_names=['train', 'valid'],
                        callbacks=callbacks,
                    )
                finally:
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stderr_fd)
                    os.close(devnull_fd)
            else:
                model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=callbacks,
                )
            
            # 保存模型
            self.models.append(model)
            
            # 预测验证集
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            # 确保 val_pred 的形状与 oof_predictions[val_idx] 匹配
            if len(val_pred.shape) == 1:
                # 如果 val_pred 是一维数组，将其扩展为二维数组
                val_pred = np.expand_dims(val_pred, axis=1)

            # 再次检查形状是否匹配
            if val_pred.shape[1] != oof_predictions.shape[1]:
                raise ValueError(
                    f"Shape mismatch: val_pred has shape {val_pred.shape}, but expected {oof_predictions[val_idx].shape}"
                )

            oof_predictions[val_idx] = val_pred
            
            # 计算验证集得分
            fold_score = log_loss(y_val, val_pred)
            cv_scores.append(fold_score)

            # 检查是否是当前最佳得分
            if fold_score < best_score:
                best_score = fold_score
                best_params = self.params.copy()
                best_fold = fold_idx + 1

            logger.info(f"Fold {fold_idx + 1} 最佳迭代: {model.best_iteration}")
            logger.info(f"Fold {fold_idx + 1} 验证得分: {fold_score:.6f}")
        
        # 计算OOF得分
        oof_score = log_loss(y, oof_predictions)
        
        logger.info("\n" + "="*60)
        logger.info("交叉验证完成!")
        logger.info(f"各折得分: {[f'{s:.6f}' for s in cv_scores]}")
        logger.info(f"平均得分: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
        logger.info(f"OOF得分: {oof_score:.6f}")
        logger.info("="*60)
        
        # 保存OOF预测
        self.oof_predictions = oof_predictions

        # 在交叉验证结束后记录最佳超参数
        logger.info("="*60)
        logger.info(f"最佳超参数出现在 Fold {best_fold}")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info("="*60)
        # 如果我们之前使用了 warnings.catch_warnings，上下文需要恢复
        if filter_context is not None:
            filter_context.__exit__(None, None, None)
        
        return oof_score
    
    def predict(self, X):
        """
        使用所有fold的模型进行预测并平均
        
        Args:
            X: 特征数组
            
        Returns:
            预测概率，形状 (n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("模型未训练，请先调用train_with_cv()")
        
        predictions = np.zeros((len(X), 7))
        
        for fold_idx, model in enumerate(self.models):
            pred = model.predict(X, num_iteration=model.best_iteration)
            predictions += pred
        
        # 平均所有fold的预测
        predictions /= len(self.models)
        
        return predictions
    
    def save_models(self, save_dir='models'):
        """
        保存所有fold的模型
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for fold_idx, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'lgb_fold_{fold_idx}.txt')
            model.save_model(model_path)
        
        logger.info(f"模型已保存到: {save_dir}")
    
    def load_models(self, save_dir='models', n_folds=None):
        """
        加载已保存的模型
        
        Args:
            save_dir: 模型目录
            n_folds: 折数（如果为None则自动检测）
        """
        if n_folds is None:
            n_folds = self.n_folds
        
        self.models = []
        
        for fold_idx in range(n_folds):
            model_path = os.path.join(save_dir, f'lgb_fold_{fold_idx}.txt')
            model = lgb.Booster(model_file=model_path)
            self.models.append(model)
        
        logger.info(f"已加载 {len(self.models)} 个模型")
    
    def get_feature_importance(self, importance_type='gain'):
        """
        获取特征重要性
        
        Args:
            importance_type: 'gain' or 'split'
            
        Returns:
            特征重要性数组
        """
        if not self.models:
            raise ValueError("模型未训练")
        
        # 平均所有fold的特征重要性
        importance = np.zeros(self.models[0].num_feature())
        
        for model in self.models:
            importance += model.feature_importance(importance_type=importance_type)
        
        importance /= len(self.models)
        
        return importance


def objective_function_factory(X, y, n_folds=5, early_stopping_rounds=50):
    """
    创建目标函数（用于HPO）
    
    这个工厂函数返回一个目标函数，可以被HPO算法调用
    
    Args:
        X: 特征数组
        y: 标签数组
        n_folds: 交叉验证折数
        early_stopping_rounds: 早停轮数
        
    Returns:
        objective_function: 接受参数字典，返回logloss的函数
    """
    def objective(params):
        """
        目标函数
        
        Args:
            params: 超参数字典
            
        Returns:
            float: 交叉验证平均logloss
        """
        # 创建训练器
        trainer = LightGBMTrainer(params=params, n_folds=n_folds)
        
        # 训练并获取CV得分
        cv_score = trainer.train_with_cv(
            X, y, 
            early_stopping_rounds=early_stopping_rounds,
            verbose=0  # 关闭详细日志
        )
        
        return cv_score
    
    return objective


def train_and_predict(
    train_features_path='data/processed/train_features.npy',
    train_labels_path='data/processed/train_labels.npy',
    test_features_path='data/processed/test_features.npy',
    test_ids_path='data/processed/test_ids.npy',
    params=None,
    n_folds=5,
    output_path='outputs/submission.csv',
    model_dir='models'
):
    """
    完整的训练和预测流程
    
    Args:
        train_features_path: 训练特征路径
        train_labels_path: 训练标签路径
        test_features_path: 测试特征路径
        test_ids_path: 测试ID路径
        params: 模型参数
        n_folds: 交叉验证折数
        output_path: 提交文件保存路径
        model_dir: 模型保存目录
    """
    logger.info("="*60)
    logger.info("开始训练和预测流程")
    logger.info("="*60)
    
    # 加载数据
    logger.info("加载数据...")
    X_train = np.load(train_features_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_features_path)
    test_ids = np.load(test_ids_path)
    
    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练模型
    trainer = LightGBMTrainer(params=params, n_folds=n_folds)
    cv_score = trainer.train_with_cv(X_train, y_train)
    
    # 保存模型
    trainer.save_models(model_dir)
    
    # 预测测试集
    logger.info("\n预测测试集...")
    test_predictions = trainer.predict(X_test)
    
    # 创建提交文件
    logger.info("创建提交文件...")
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
    
    # 保存提交文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    logger.info(f"提交文件已保存到: {output_path}")
    logger.info(f"CV得分: {cv_score:.6f}")
    logger.info("="*60)
    
    return trainer, cv_score


if __name__ == '__main__':
    # 直接运行此脚本进行训练和预测
    train_and_predict()

