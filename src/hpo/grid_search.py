"""
Grid Search - 网格搜索
完整实现版本

注意事项:
1. Grid Search会遍历所有参数组合
2. 对于连续参数，已选择3-5个代表性的值
3. 对于SVM和MLP，已减少网格点数以确保可以跑完
"""

import numpy as np
import logging
from tqdm import tqdm
from itertools import product
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class GridSearch(BaseHPO):
    """
    Grid Search - 网格搜索超参数优化
    
    遍历所有预定义的参数组合，找到最优解
    """
    
    def __init__(self, model_name='lightgbm', n_trials=None, cv_folds=5, random_state=42):
        """
        初始化Grid Search
        
        Args:
            model_name: 模型名称 ('lightgbm', 'svm', 'mlp')
            n_trials: 搜索次数（Grid Search不需要，自动计算）
            cv_folds: 交叉验证折数（默认5）
            random_state: 随机种子（默认42）
        
        注意: Grid Search不需要n_trials参数，因为它会遍历所有组合
        """
        # Grid Search的n_trials由网格大小决定，这里设为0占位
        super().__init__(model_name, n_trials=0, cv_folds=cv_folds, random_state=random_state)
        
        # 定义离散化的搜索网格
        self.param_grid = self._create_param_grid()
        self.total_grid_size = self._count_grid_size()
        
        # 根据用户的n_trials选择是否子采样网格
        self.param_combinations = self._prepare_param_combinations(n_trials)
        
        if self.total_grid_size == self.n_trials:
            logger.info(f"网格搜索将尝试全部 {self.n_trials} 种参数组合")
        else:
            logger.info(
                f"网格搜索原有 {self.total_grid_size} 种组合，本次根据n_trials采样 {self.n_trials} 种"
            )
    
    def _create_param_grid(self):
        """
        创建离散化的参数网格
        
        Returns:
            dict: 参数网格，每个参数对应一个值列表
        """
        param_grid = {}
        
        if self.model_name == 'lightgbm':
            # LightGBM: 可以有较多组合（3*3*3*3*2*2 = 324）
            param_grid = {
                'num_leaves': [31, 63, 100],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [200, 500, 800],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.7, 0.9],
            }
        
        elif self.model_name == 'svm':
            # SVM: 减少组合（3*2*3 = 18）
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': [0.001, 0.01, 0.1],
            }
        
        elif self.model_name == 'mlp':
            # MLP: 减少组合（3*2*2*2 = 24）
            param_grid = {
                'hidden_layer_sizes': [(128,), (256,), (256, 128)],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001],
                'batch_size': [64, 128],
            }
        
        return param_grid
    
    def _count_grid_size(self):
        """计算网格大小（总试验次数）"""
        count = 1
        for param_values in self.param_grid.values():
            count *= len(param_values)
        return count
    
    def _generate_all_combinations(self):
        """
        生成所有参数组合
        
        Yields:
            dict: 参数字典
        """
        # 获取所有参数名和对应的值列表
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        
        # 生成所有组合
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            yield params

    def _prepare_param_combinations(self, requested_trials):
        """根据n_trials决定实际需要评估的网格点"""
        all_combinations = list(self._generate_all_combinations())
        grid_size = len(all_combinations)
        
        # 没有限制或限制大于等于网格尺寸 -> 评估全部组合
        if not requested_trials or requested_trials >= grid_size:
            self.n_trials = grid_size
            return all_combinations
        
        # 按随机种子采样部分组合，保证可重复
        rng = np.random.default_rng(self.random_state)
        selected_indices = rng.choice(grid_size, size=requested_trials, replace=False)
        selected_indices.sort()
        self.n_trials = requested_trials
        return [all_combinations[idx] for idx in selected_indices]
    
    def optimize(self, objective_function, verbose=True):
        """
        执行Grid Search优化
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回logloss
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        logger.info(f"开始Grid Search，共 {self.n_trials} 种参数组合")
        logger.info(f"模型: {self.model_name}, 交叉验证: {self.cv_folds} 折")
        logger.info(f"参数网格: {self.param_grid}")
        logger.info("="*60)
        
        best_score = float('inf')
        best_params = None
        
        # 遍历所有参数组合
        iterator = tqdm(
            enumerate(self.param_combinations), 
            total=self.n_trials,
            desc="Grid Search进度"
        ) if verbose else enumerate(self.param_combinations)
        
        for trial_idx, params in iterator:
            try:
                # 评估超参数
                score = objective_function(params)
                
                # 记录历史
                self.history.append({
                    'trial': trial_idx,
                    'params': params,
                    'score': score
                })
                
                # 更新最佳结果
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    
                    if verbose:
                        logger.info(f"\n✓ 发现更好的参数! Trial {trial_idx}, Score: {score:.6f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} 失败: {str(e)}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("Grid Search完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info(f"最佳参数: {best_params}")
        logger.info("="*60)
        
        return best_params, best_score

