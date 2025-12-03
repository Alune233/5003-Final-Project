import logging
from itertools import product, islice

import numpy as np
from tqdm import tqdm

from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)


class GridSearch(BaseHPO):

    def __init__(self, n_trials=50, cv_folds=5, random_state=42, param_grid=None):
        super().__init__(n_trials, cv_folds, random_state)
        self.param_grid = param_grid or self._default_grid()
        self._param_names = list(self.param_grid.keys())

        grid_lengths = [len(self.param_grid[name]) for name in self._param_names]
        self.total_combinations = int(np.prod(grid_lengths)) if grid_lengths else 0

        if self.total_combinations == 0:
            raise ValueError("GridSearch参数网格为空")

        original_trials = self.n_trials
        self.n_trials = min(self.n_trials, self.total_combinations)

        if original_trials > self.n_trials:
            logger.info(
                "请求的n_trials=%d超过网格组合数%d，将仅评估前%d个组合",
                original_trials,
                self.total_combinations,
                self.n_trials,
            )

        self._grid_iterator = product(*(self.param_grid[name] for name in self._param_names))
        self._consumed = 0

        logger.info(
            "GridSearch初始化完成，网格大小=%d，实际评估=%d",
            self.total_combinations,
            self.n_trials,
        )

    def _default_grid(self):
        return {
            'num_leaves': [31, 63, 127],
            'max_depth': [6, 9],
            'learning_rate': [0.03, 0.05],
            'n_estimators': [300, 600],
            'min_child_samples': [20, 40],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0],
            'reg_lambda': [1.0],
            'min_split_gain': [0.0],
        }

    def suggest_params(self):
        if self._consumed >= self.n_trials:
            raise StopIteration("GridSearch已经遍历完指定的组合数")

        try:
            combination = next(self._grid_iterator)
        except StopIteration as exc:
            raise StopIteration("GridSearch参数组合耗尽") from exc

        params = dict(zip(self._param_names, combination))
        params.update(self.fixed_params)

        self._consumed += 1
        return params

    def optimize(self, objective_function, verbose=True):
        logger.info("开始Grid Search，共%d个组合", self.n_trials)
        logger.info("交叉验证: %d 折", self.cv_folds)
        logger.info("=" * 60)

        best_score = float('inf')
        best_params = None

        iterator = tqdm(range(self.n_trials), desc="Grid Search进度") if verbose else range(self.n_trials)

        for trial_idx in iterator:
            try:
                params = self.suggest_params()
            except StopIteration:
                logger.info("参数组合提前耗尽，结束搜索")
                break

            try:
                score = objective_function(params)

                self.history.append({
                    'trial': trial_idx,
                    'params': params,
                    'score': score,
                })

                if score < best_score:
                    best_score = score
                    best_params = params.copy()

                    if verbose:
                        logger.info(
                            "\n发现更优参数! Trial %d\n   得分: %.6f\n   num_leaves: %s, max_depth: %s, lr: %.3f, n_estimators: %s",
                            trial_idx,
                            score,
                            params.get('num_leaves'),
                            params.get('max_depth'),
                            params.get('learning_rate'),
                            params.get('n_estimators'),
                        )

            except Exception as exc:
                logger.error("Trial %d 失败: %s", trial_idx, exc)
                continue

        logger.info("\n" + "=" * 60)
        logger.info("Grid Search完成!")
        logger.info("最佳得分: %.6f", best_score)

        if best_params:
            logger.info("最佳参数:")
            for key, value in best_params.items():
                if key not in self.fixed_params:
                    logger.info("  %s: %s", key, value)

        logger.info("=" * 60)

        return best_params, best_score

    def plot_optimization_history(self, save_path=None):
        try:
            import matplotlib.pyplot as plt

            if not self.history:
                logger.warning("没有历史记录可以绘制")
                return

            trials = [h['trial'] for h in self.history]
            scores = [h['score'] for h in self.history]

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
            plt.title('Grid Search Optimization History')
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
                logger.info("优化历史图已保存到: %s", save_path)
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")
