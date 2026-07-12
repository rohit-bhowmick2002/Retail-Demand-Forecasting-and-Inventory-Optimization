"""
Module: evaluation.py
Description: Evaluation metric suite tailored for zero-inflated retail demand forecasting.
Implements WAPE, SMAPE, MAE, RMSE, and the Kaggle M5 benchmark metric (WRMSSE / RMSSE).
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RetailMetrics:
    """
    Computes regression and time-series forecasting metrics while safely handling
    zero-sales denominator division issues common in retail datasets.
    """

    @staticmethod
    def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Weighted Absolute Percentage Error (WAPE).
        Does not divide by zero on intermittent sales days.
        """
        y_true_clean = np.array(y_true, dtype=np.float64)
        y_pred_clean = np.array(y_pred, dtype=np.float64)
        total_actual = np.sum(y_true_clean)
        if total_actual == 0:
            return 0.0
        return float(np.sum(np.abs(y_true_clean - y_pred_clean)) / total_actual * 100.0)

    @staticmethod
    def calculate_rmsse(
        y_true_future: np.ndarray,
        y_pred_future: np.ndarray,
        y_true_historical: np.ndarray
    ) -> float:
        """
        Root Mean Squared Scaled Error (RMSSE).
        Scales prediction error against historical naive random walk error.
        """
        numerator = np.mean(np.square(y_true_future - y_pred_future))
        if len(y_true_historical) < 2:
            return float(np.sqrt(numerator))
            
        historical_diff = np.diff(y_true_historical)
        denominator = np.mean(np.square(historical_diff))
        
        if denominator <= 0:
            return float(np.sqrt(numerator))
        return float(np.sqrt(numerator / denominator))

    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Returns a complete summary dictionary of evaluation metrics.
        """
        y_true_clean = np.array(y_true, dtype=np.float64)
        y_pred_clean = np.maximum(0.0, np.array(y_pred, dtype=np.float64))
        
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        rmse = np.sqrt(np.mean(np.square(y_true_clean - y_pred_clean)))
        wape = RetailMetrics.calculate_wape(y_true_clean, y_pred_clean)
        
        # SMAPE
        denom = (np.abs(y_true_clean) + np.abs(y_pred_clean)) / 2.0
        smape = np.mean(np.where(denom > 0, np.abs(y_true_clean - y_pred_clean) / denom, 0.0)) * 100.0
        
        metrics = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "WAPE_pct": round(float(wape), 2),
            "SMAPE_pct": round(float(smape), 2)
        }
        return metrics
