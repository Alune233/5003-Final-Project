from .base_hpo import BaseHPO
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class SMAC3HPO(BaseHPO):
    def __init__(self, n_trials=50, cv_folds=5, random_state=42, use_history=False, history_factor=0.2, initial_history_path=None):
        super().__init__(n_trials, cv_folds, random_state)
        # If use_history==True and we have history, we'll shrink search space
        self.use_history = use_history
        self.history_factor = history_factor
        self.initial_history_path = initial_history_path

        # 为与 RandomSearch 对齐，也使用 search_space 字典定义搜索空间
        self.search_space = {
            'num_leaves': ('int', 20, 150),
            'learning_rate': ('float_log', 0.01, 0.3),
            'max_depth': ('int', 3, 12),
            'n_estimators': ('int', 100, 1000),
            'min_child_samples': ('int', 10, 100),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'reg_alpha': ('float', 0.0, 10.0),
            'reg_lambda': ('float', 0.0, 10.0),
            'min_split_gain': ('float', 0.0, 1.0),
        }

        # 将 search_space 写回 BaseHPO.search_space
        self.search_space = self.search_space
        self.cs = None

    def suggest_params(self):
        # Reserved: 通过简单的随机采样给出建议（主要用于测试或兼容性）
        params = {}
        for name, spec in self.search_space.items():
            p_type, low, high = spec
            if p_type == 'int':
                params[name] = int(np.random.randint(int(low), int(high) + 1))
            elif p_type == 'float':
                params[name] = float(np.random.uniform(float(low), float(high)))
            elif p_type == 'float_log':
                log_low = np.log(float(low))
                log_high = np.log(float(high))
                params[name] = float(np.exp(np.random.uniform(log_low, log_high)))
        params.update(self.fixed_params)
        return params

    def optimize(self, objective_function, verbose=True) -> Tuple[dict, float]:
        # 如果提供初始history文件，先加载
        if self.initial_history_path:
            try:
                self.load_history(self.initial_history_path)
                logger.info(f"Loaded initial HPO history from: {self.initial_history_path}")
            except Exception as e:
                logger.warning(f"无法加载初始历史: {e}")

        # 如果需要从历史中缩小搜索空间，则先进行更新
        if self.use_history and self.history:
            self.update_search_space_around_best(factor=self.history_factor)

        # Build ConfigurationSpace from self.search_space
        self.cs = ConfigurationSpace()
        for name, spec in self.search_space.items():
            p_type, low, high = spec
            if p_type == 'int':
                self.cs.add_hyperparameter(UniformIntegerHyperparameter(name, lower=int(low), upper=int(high)))
            elif p_type == 'float':
                self.cs.add_hyperparameter(UniformFloatHyperparameter(name, lower=float(low), upper=float(high)))
            elif p_type == 'float_log':
                # 对数尺度参数
                self.cs.add_hyperparameter(UniformFloatHyperparameter(name, lower=float(low), upper=float(high), log=True))
            else:
                # 其他类型忽略
                continue

        # Define the SMAC scenario
        # SMAC 2.x Scenario
        scenario = Scenario(
            self.cs,
            deterministic=True,
            n_trials=self.n_trials,
        )

        # Wrapper for the objective function to handle Configuration object
        def smac_objective(config, seed=0):
            try:
                cfg = config.get_dictionary()
                # 将整数参数强制转换为 int
                for name, spec in self.search_space.items():
                    if spec[0] == 'int' and name in cfg:
                        try:
                            cfg[name] = int(round(float(cfg[name])))
                        except Exception:
                            cfg[name] = int(cfg[name])
                result = objective_function(cfg)
                # Ensure a float is returned
                return float(result)
            except Exception as e:
                logger.error(f"SMAC objective crashed: {e}")
                return float('inf')

        # Define the SMAC optimizer using the Facade
        smac = HyperparameterOptimizationFacade(
            scenario,
            smac_objective,
            overwrite=True,
        )

        # Perform the optimization
        incumbent = smac.optimize()

        # Retrieve the best configuration and its score
        best_params = incumbent.get_dictionary()
        best_score = smac.runhistory.get_cost(incumbent)

        # Save the history in the same format as RandomSearch: {'trial', 'params', 'score'}
        self.history = []
        try:
            # runhistory is dict-like: key -> run info
            trial_idx = 0
            for key, value in smac.runhistory.items():
                params = None
                try:
                    config_id = key.config_id
                    if config_id in smac.runhistory.ids_config:
                        config = smac.runhistory.ids_config[config_id]
                        params = config.get_dictionary()
                except Exception:
                    # fallback: attempt to extract from key
                    try:
                        params = key.config.get_dictionary()
                    except Exception:
                        params = None

                score = float(getattr(value, 'cost', float('inf')))
                self.history.append({'trial': trial_idx, 'params': params, 'score': score})
                trial_idx += 1
        except Exception as e:
            logger.warning(f"无法完全解析 SMAC 运行历史: {e}")

        # Log best result similar to RandomSearch
        logger.info("\n" + "="*60)
        logger.info("SMAC3 优化完成!")
        logger.info(f"最佳得分: {best_score:.6f}")
        logger.info("最佳参数:")
        for key, value in best_params.items():
            if key not in self.fixed_params:
                logger.info(f"  {key}: {value}")
        logger.info("="*60)

        return best_params, best_score