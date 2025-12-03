import numpy as np
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHPO(ABC):
    """
    HPO算法的抽象基类
    
    子类需要实现:
        - suggest_params(): 建议下一组超参数
        - optimize(): 执行优化过程
    """
    
    def __init__(self, n_trials=50, cv_folds=5, random_state=42):
        """
        初始化HPO算法
        
        Args:
            n_trials: 搜索次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # LightGBM固定参数（所有算法共享）
        self.fixed_params = {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'verbosity': -1,
            # Some LGB builds will decide on columnwise histogram threading automatically – forcing removes info message
            'force_col_wise': True,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # 存储历史结果
        self.history = []
        # 通用搜索空间定义，子类可以在初始化时覆盖
        # 格式与 RandomSearch.search_space 保持一致: {'param': ('type', low, high), ...}
        self.search_space = {}
        
        logger.info(f"初始化 {self.__class__.__name__}")
        logger.info(f"n_trials={n_trials}, cv_folds={cv_folds}, random_state={random_state}")
    
    @abstractmethod
    def suggest_params(self):
        """
        建议下一组超参数
        
        Returns:
            dict: 超参数字典
        """
        pass
    
    @abstractmethod
    def optimize(self, objective_function, verbose=True):
        """
        执行优化过程
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回评分
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (最佳参数, 最佳得分)
        """
        pass
    
    def get_best_params(self):
        """
        获取历史最佳参数
        
        Returns:
            dict: 最佳参数字典
        """
        if not self.history:
            logger.warning("没有历史记录")
            return None
        
        # 找到得分最低的（logloss越小越好）
        best_idx = np.argmin([h['score'] for h in self.history])
        return self.history[best_idx]['params']
    
    def get_best_score(self):
        """
        获取历史最佳得分
        
        Returns:
            float: 最佳得分
        """
        if not self.history:
            logger.warning("没有历史记录")
            return None
        
        return min([h['score'] for h in self.history])
    
    def save_history(self, save_path):
        """
        保存优化历史
        
        Args:
            save_path: 保存路径
        """
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"优化历史已保存到: {save_path}")

    def load_history(self, save_path):
        """
        从文件加载优化历史
        """
        import json
        with open(save_path, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        logger.info(f"优化历史已从 {save_path} 加载")

    def plot_optimization_history(self, save_path=None):
        """
        默认绘制优化历史记录（history）: 都为 [ {'trial', 'params', 'score'} ]
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("无法加载 matplotlib，无法绘制优化历史")
            return

        if not self.history:
            logger.warning("没有历史记录可以绘制")
            return

        trials = [h.get('trial', i) for i, h in enumerate(self.history)]
        scores = [h.get('score', float('nan')) for h in self.history]

        # 计算累积最佳得分
        best_scores = []
        current_best = float('inf')
        for score in scores:
            current_best = min(current_best, score)
            best_scores.append(current_best)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(trials, scores, alpha=0.6, s=30)
        plt.plot(trials, best_scores, 'r-', linewidth=2, label='Best Score')
        plt.xlabel('Trial')
        plt.ylabel('Score (logloss)')
        plt.title(f'{self.__class__.__name__} Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(min(scores), color='r', linestyle='--', linewidth=2, label=f'Best: {min(scores):.6f}')
        plt.xlabel('Score (logloss)')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"优化历史图已保存到: {save_path}")
        else:
            plt.show()

    def update_search_space_around_best(self, factor=0.2, min_int_width=1, min_float_width=1e-6):
        """
        基于历史最佳参数，将搜索空间缩小到以最佳参数为中心的范围。

        Args:
            factor: 缩放因子（例如 0.2 表示 +/-20%）
            min_int_width: 整数参数允许的最小区间宽度
            min_float_width: 浮点参数允许的最小区间宽度
        """
        if not self.history:
            logger.warning("没有历史记录可用于更新搜索空间")
            return

        best = self.get_best_params()
        if not best:
            logger.warning("无法从历史中获取最佳参数")
            return

        new_space = {}
        for name, spec in self.search_space.items():
            try:
                p_type, low, high = spec
            except Exception:
                # 非预期格式，跳过
                new_space[name] = spec
                continue

            if name not in best:
                new_space[name] = spec
                continue

            best_val = best[name]

            if p_type == 'int':
                total_width = max(high - low, 1)
                half_width = max(int(total_width * factor / 2), min_int_width)
                new_low = max(low, int(best_val - half_width))
                new_high = min(high, int(best_val + half_width))
                if new_low >= new_high:
                    new_low = max(low, int(best_val - min_int_width))
                    new_high = min(high, int(best_val + min_int_width))
                new_space[name] = ('int', new_low, new_high)

            elif p_type in ('float', 'float_log'):
                # 对数尺度参数使用乘法缩放
                if p_type == 'float_log' and best_val > 0:
                    new_low = max(low, best_val * (1 - factor))
                    new_high = min(high, best_val * (1 + factor))
                else:
                    total_width = max(high - low, min_float_width)
                    half_width = max(total_width * factor / 2, min_float_width)
                    new_low = max(low, best_val - half_width)
                    new_high = min(high, best_val + half_width)
                if new_low >= new_high:
                    new_low = max(low, best_val - min_float_width)
                    new_high = min(high, best_val + min_float_width)
                new_space[name] = (p_type, float(new_low), float(new_high))

            else:
                # 不支持的类型，直接保留
                new_space[name] = spec

        logger.info("已根据历史最佳参数缩小搜索空间")
        self.search_space = new_space
