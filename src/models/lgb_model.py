import os
import numpy as np
import lightgbm as lgb
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM模型"""
    
    def _create_model(self, params):
        """创建LightGBM模型"""
        # 合并固定参数
        cpu_threads = max(1, (os.cpu_count() or 1) - 1)
        model_params = {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': cpu_threads,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,
            'force_col_wise': True,
            'max_bin': 127,
            'num_threads': cpu_threads,
            **params  # 用户参数
        }
        return model_params
    
    def _fit(self, model_params, X_train, y_train, X_val=None, y_val=None):
        """训练LightGBM模型"""
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32, copy=False)
        if X_val is not None and X_val.dtype != np.float32:
            X_val = X_val.astype(np.float32, copy=False)

        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        model = lgb.train(
            model_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # 关闭日志
            ]
        )
        
        return model
    
    def _predict_proba(self, model, X):
        """预测概率"""
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        return model.predict(X, num_iteration=model.best_iteration)

