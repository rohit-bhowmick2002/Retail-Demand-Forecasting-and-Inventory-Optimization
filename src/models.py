"""
Module: models.py
Description: Global cross-sectional machine learning forecasting engine.
Supports LightGBM with Tweedie objective (`tweedie_variance_power=1.1`) specifically
tailored for zero-inflated, intermittent retail time series.
"""

import logging
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GlobalDemandForecaster:
    """
    Unified global time-series forecaster training across multiple SKUs and store locations simultaneously.
    Overcomes the '300,000 models' trap by learning shared elasticity and seasonal patterns.
    """

    def __init__(self, use_tweedie: bool = True, tweedie_power: float = 1.1):
        self.use_tweedie = use_tweedie
        self.tweedie_power = tweedie_power
        self.model = None
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    def get_default_params(self) -> Dict[str, Any]:
        """
        Returns optimized LightGBM hyperparameter configuration for M5 benchmark.
        """
        params = {
            "objective": "tweedie" if self.use_tweedie else "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 1200,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_child_samples": 45,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        if self.use_tweedie:
            params["tweedie_variance_power"] = self.tweedie_power
        return params

    def train_time_series_split(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        val_days: int = 28
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Splits data temporally (all data before `max_date - val_days` is Train, final `val_days` is Validation).
        Trains Global LightGBM model and evaluates forecast performance.
        """
        logger.info(f"Executing temporal train-validation split (Validation Horizon = last {val_days} days)...")
        
        # Prepare feature column list
        ignore_cols = {"id", "date", "d", target_col, "sales", "has_sale", "rolling_max_price"}
        self.feature_cols = [c for c in df.columns if c not in ignore_cols]
        
        max_date = df["date"].max()
        val_start_date = max_date - pd.Timedelta(days=val_days)
        
        train_df = df[df["date"] <= val_start_date].dropna(subset=self.feature_cols + [target_col])
        val_df = df[df["date"] > val_start_date].dropna(subset=self.feature_cols + [target_col])
        
        logger.info(f"Training set: {train_df.shape[0]:,} rows | Validation set: {val_df.shape[0]:,} rows")
        
        X_train, y_train = train_df[self.feature_cols], train_df[target_col]
        X_val, y_val = val_df[self.feature_cols], val_df[target_col]
        
        if LIGHTGBM_AVAILABLE:
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            self.model = self._train_sklearn_fallback(X_train, y_train)
            
        # Evaluate validation metrics
        val_preds = self.predict(X_val)
        val_preds = np.maximum(0.0, val_preds)  # Demand cannot be negative
        
        mae = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        wape = np.sum(np.abs(y_val - val_preds)) / np.sum(y_val) * 100.0 if np.sum(y_val) > 0 else 0.0
        
        metrics = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "WAPE_pct": round(float(wape), 2)
        }
        logger.info(f"Model Evaluation Results on Validation Horizon: {metrics}")
        return self.model, metrics

    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        logger.info("Training Global LightGBM forecaster with Tweedie objective loss...")
        params = self.get_default_params()
        
        # Convert categorical columns for LightGBM
        for c in self.categorical_cols:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category")
                X_val[c] = X_val[c].astype("category")
                
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        return model

    def _train_sklearn_fallback(self, X_train, y_train):
        logger.info("LightGBM not installed. Using HistGradientBoostingRegressor fallback...")
        X_train = X_train.copy()
        for c in self.categorical_cols:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category").cat.codes
                
        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates non-negative unit sales predictions for future time horizons.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_time_series_split() first.")
            
        X_infer = X[self.feature_cols].copy()
        if LIGHTGBM_AVAILABLE:
            for c in self.categorical_cols:
                if c in X_infer.columns:
                    X_infer[c] = X_infer[c].astype("category")
        else:
            for c in self.categorical_cols:
                if c in X_infer.columns:
                    X_infer[c] = X_infer[c].astype("category").cat.codes
                    
        preds = self.model.predict(X_infer)
        return np.maximum(0.0, preds)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns ranked feature importance based on model split/gain metrics.
        """
        if not LIGHTGBM_AVAILABLE or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
            
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        
        return importance_df
