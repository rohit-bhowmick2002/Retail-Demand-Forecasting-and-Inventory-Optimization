"""
Module: inventory_engine.py
Description: Mathematical Operations Research (OR) supply chain engine.
Converts ML demand predictions and demand uncertainty (sigma_d) into actionable
Safety Stock (SS), Reorder Points (ROP), Economic Order Quantity (EOQ), and Risk Alerts.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    Executes core supply chain formulas to optimize warehouse inventory holding costs
    while preventing out-of-stock lost sales.
    """

    def __init__(
        self,
        lead_time_days: float = 7.0,
        lead_time_std_days: float = 1.5,
        service_level_z: float = 1.65,  # 1.65 ≈ 95% service level
        holding_cost_rate: float = 0.25, # 25% annual carrying cost
        fixed_order_cost: float = 15.0   # $15 fixed cost to issue a purchase order
    ):
        self.lead_time_days = lead_time_days
        self.lead_time_std_days = lead_time_std_days
        self.service_level_z = service_level_z
        self.holding_cost_rate = holding_cost_rate
        self.fixed_order_cost = fixed_order_cost

    def calculate_inventory_policy(self, forecast_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates complete supply chain policy matrix for each SKU-Store.
        Required input columns: ['store_id', 'item_id', 'sell_price', 'pred_mean_demand', 'pred_std_demand']
        Optional input column: ['current_stock']
        """
        logger.info("Calculating Safety Stock, Reorder Points (ROP), and EOQ across all SKU-Stores...")
        df = forecast_summary_df.copy()
        
        # 1. Lead Time Demand (LTD)
        df["lead_time_demand"] = df["pred_mean_demand"] * self.lead_time_days
        
        # 2. Dynamic Safety Stock (SS) accounting for both demand variance and supplier lead time variance
        # Formula: SS = Z * sqrt( (LT * std_demand^2) + (mean_demand^2 * std_LT^2) )
        variance_term = (
            (self.lead_time_days * (df["pred_std_demand"] ** 2)) +
            ((df["pred_mean_demand"] ** 2) * (self.lead_time_std_days ** 2))
        )
        df["safety_stock"] = np.ceil(self.service_level_z * np.sqrt(np.maximum(0.0, variance_term)))
        
        # 3. Reorder Point (ROP)
        # When warehouse stock reaches ROP, place an automated purchase order
        df["reorder_point"] = np.ceil(df["lead_time_demand"] + df["safety_stock"])
        
        # 4. Economic Order Quantity (EOQ)
        # Formula: EOQ = sqrt( (2 * Annual_Demand * Order_Cost) / Holding_Cost )
        annual_demand = df["pred_mean_demand"] * 365.0
        unit_holding_cost = np.where(df["sell_price"] * self.holding_cost_rate > 0, df["sell_price"] * self.holding_cost_rate, 0.50)
        
        eoq_raw = np.sqrt((2.0 * annual_demand * self.fixed_order_cost) / unit_holding_cost)
        df["eoq_reorder_qty"] = np.ceil(np.maximum(1.0, eoq_raw))
        
        # 5. Inventory Risk Classification
        if "current_stock" not in df.columns:
            # Simulate realistic current stock on hand for evaluation
            np.random.seed(42)
            df["current_stock"] = np.round(df["reorder_point"] * np.random.uniform(0.3, 3.8, size=len(df)))
            
        df["status"] = "OPTIMAL: Balanced Inventory"
        
        # Critical stockout risk condition
        crit_mask = df["current_stock"] <= df["reorder_point"]
        df.loc[crit_mask, "status"] = "CRITICAL: Stockout Risk (Order Now)"
        
        # Overstocked warning condition
        over_mask = df["current_stock"] > (df["reorder_point"] + (2.5 * df["eoq_reorder_qty"]))
        df.loc[over_mask, "status"] = "WARNING: Overstocked (Discount/Stop Order)"
        
        # 6. Financial Impact Metrics
        # Potential stockout loss if not ordered = 30 days demand * price * 0.95 margins
        df["prevented_stockout_loss_usd"] = np.where(
            crit_mask,
            df["pred_mean_demand"] * 30.0 * df["sell_price"] * 0.95,
            0.0
        )
        
        # Excess working capital locked in overstock = (current_stock - target_max) * price
        df["overstock_capital_locked_usd"] = np.where(
            over_mask,
            (df["current_stock"] - (df["reorder_point"] + df["eoq_reorder_qty"])) * df["sell_price"],
            0.0
        )
        df["annual_holding_savings_usd"] = df["overstock_capital_locked_usd"] * self.holding_cost_rate
        
        logger.info("Supply chain inventory optimization calculations completed.")
        return df

    def calculate_newsvendor_fractile(self, unit_price: float, unit_cost: float, salvage_value: float = 0.0) -> float:
        """
        Calculates the optimal Newsvendor critical fractile for perishable / single-period seasonal items.
        Formula: CF = (Price - Cost) / (Price - Salvage_Value)
        """
        underage_cost = unit_price - unit_cost
        overage_cost = unit_cost - salvage_value
        if underage_cost + overage_cost <= 0:
            return 0.50
        return round(underage_cost / (underage_cost + overage_cost), 4)
