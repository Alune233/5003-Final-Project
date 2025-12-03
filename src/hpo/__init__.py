from .base_hpo import BaseHPO
from .random_search import RandomSearch
from .smac3_hpo import SMAC3HPO

AVAILABLE_ALGORITHMS = {
    'random': RandomSearch,
    'smac3': SMAC3HPO,  # Add SMAC3 algorithm
    # 'bayesian': BayesianOptimization,  # 队友添加示例
    # 'tpe': TPEOptimizer,              # 队友添加示例
    # 'optuna': OptunaOptimizer,        # 队友添加示例
}


def get_hpo_algorithm(algorithm_name):
    """
    获取HPO算法类
    
    Args:
        algorithm_name: 算法名称（如 'random', 'bayesian'）
        
    Returns:
        HPO算法类
        
    Raises:
        ValueError: 如果算法名称不存在
    """
    if algorithm_name not in AVAILABLE_ALGORITHMS:
        available = ', '.join(AVAILABLE_ALGORITHMS.keys())
        raise ValueError(
            f"未知的HPO算法: '{algorithm_name}'\n"
            f"可用的算法: {available}\n"
            f"如需添加新算法，请在 src/hpo/__init__.py 的 AVAILABLE_ALGORITHMS 中注册"
        )
    return AVAILABLE_ALGORITHMS[algorithm_name]


def list_algorithms():
    """列出所有可用的HPO算法"""
    return list(AVAILABLE_ALGORITHMS.keys())


__all__ = ['BaseHPO', 'RandomSearch', 'get_hpo_algorithm', 'list_algorithms', 'AVAILABLE_ALGORITHMS']

