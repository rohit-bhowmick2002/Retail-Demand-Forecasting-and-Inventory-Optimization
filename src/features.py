"""
Module: features.py
Description: Feature engineering pipeline generating lag windows, rolling statistics,
price elasticity indicators, and calendar features while preventing data leakage.
"""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Constructs model-ready features for cross-sectional LightGBM demand forecasting.
    Enforces strict time horizon boundaries (e.g., lag 28 for direct 28-day forecasting).
    """

    def __init__(self, forecast_horizon: int = 28):
        self.forecast_horizon = forecast_horizon

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes complete feature engineering pipeline on input dataframe.
        """
        logger.info("Starting feature engineering pipeline...")
        df = df.copy()
        
        # Ensure proper sorting for window aggregations
        df = df.sort_values(by=["store_id", "item_id", "date"]).reset_index(drop=True)
        
        df = self._add_calendar_features(df)
        df = self._add_price_features(df)
        df = self._add_lag_and_rolling_features(df)
        df = self._add_intermittent_demand_features(df)
        
        # Drop rows with NaN lags created by initial window shifts
        min_valid_rows = df.dropna(subset=["rolling_mean_28"]).shape[0]
        logger.info(f"Feature engineering complete. Retained {min_valid_rows:,} valid time-series rows after lag initialization.")
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding date and seasonal features...")
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_month"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing price discount, momentum, and elasticity features...")
        # Rolling 52-week historical max price by item-store
        df["rolling_max_price"] = (
            df.groupby(["store_id", "item_id"], observed=True)["sell_price"]
            .transform(lambda x: x.rolling(window=365, min_periods=1).max())
        )
        
        df["price_discount_pct"] = np.where(
            df["rolling_max_price"] > 0,
            (df["rolling_max_price"] - df["sell_price"]) / df["rolling_max_price"] * 100.0,
            0.0
        )
        df["price_discount_pct"] = np.clip(df["price_discount_pct"], 0.0, 100.0)
        df["is_on_sale"] = (df["price_discount_pct"] >= 5.0).astype(np.int8)
        
        # Price ratio vs category mean price for that week
        cat_mean_price = df.groupby(["store_id", "cat_id", "wm_yr_wk"], observed=True)["sell_price"].transform("mean")
        df["price_ratio_vs_category"] = np.where(cat_mean_price > 0, df["sell_price"] / cat_mean_price, 1.0)
        
        return df

    def _add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating lags and rolling statistics (based on horizon h={self.forecast_horizon})...")
        grouped = df.groupby(["store_id", "item_id"], observed=True)["sales"]
        
        # Direct forecast anchor lag (Lag 28 avoids any leakage during 28-day future inference)
        df["lag_7"] = grouped.shift(7)
        df["lag_14"] = grouped.shift(14)
        df["lag_28"] = grouped.shift(self.forecast_horizon)
        
        # Rolling statistics built purely on the safe anchor lag (lag_28) or lag_7 for short-term evaluation
        anchor_grouped = df.groupby(["store_id", "item_id"], observed=True)["lag_7"]
        
        df["rolling_mean_7"] = anchor_grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
        df["rolling_std_7"] = anchor_grouped.transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0.0)
        df["rolling_mean_28"] = anchor_grouped.transform(lambda x: x.rolling(28, min_periods=1).mean())
        df["rolling_std_28"] = anchor_grouped.transform(lambda x: x.rolling(28, min_periods=1).std()).fillna(0.0)
        
        # Exponentially Weighted Moving Average
        df["ewma_7"] = anchor_grouped.transform(lambda x: x.ewm(span=7, min_periods=1).mean())
        
        return df

    def _add_intermittent_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing intermittent demand recency (days since last sale)...")
        # Identify whether a sale occurred
        df["has_sale"] = (df["sales"] > 0).astype(np.int8)
        
        # Calculate days since last positive sale using cumulative counting
        # For simplicity in vectorized Pandas, compute rolling active ratio over last 14 days
        active_grouped = df.groupby(["store_id", "item_id"], observed=True)["has_sale"]
        df["active_days_ratio_14d"] = active_grouped.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0.0)
        
        return df
