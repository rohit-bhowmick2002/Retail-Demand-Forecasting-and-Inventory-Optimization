"""
Script: run_end_to_end_pipeline.py
Description: End-to-End verification script that executes Data Loading, Feature Engineering,
Global ML Forecasting, Hierarchical Reconciliation, and Operations Research Supply Chain Optimization.
Outputs summary tables and metrics to verify project integrity.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import RetailDataLoader
from features import FeatureEngineer
from models import GlobalDemandForecaster
from reconciliation import HierarchicalReconciler
from inventory_engine import InventoryOptimizer
from evaluation import RetailMetrics

def main():
    print("=====================================================================================")
    print("AI-POWERED RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION ENGINE - END-TO-END RUN")
    print("=====================================================================================")
    
    # Step 1: Ingest and Downcast Data
    loader = RetailDataLoader(data_dir="data/raw")
    raw_df = loader.load_and_downcast(sample_skus=150)
    
    if hasattr(raw_df, "to_pandas"):
        df = raw_df.to_pandas()
    else:
        df = raw_df
        
    print(f"[Step 1] Data Ingestion & Downcasting Complete: {df.shape[0]:,} rows | {df.shape[1]} columns.")
    
    # Save a sample to data/sample for GitHub reviewers
    os.makedirs("data/sample", exist_ok=True)
    sample_export = df.head(5000).copy()
    sample_export.to_csv("data/sample/m5_synthetic_sample.csv", index=False)
    print(f"[Step 1] Exported sample dataset to data/sample/m5_synthetic_sample.csv ({len(sample_export):,} rows).")
    
    # Step 2: Feature Engineering
    fe = FeatureEngineer(forecast_horizon=28)
    features_df = fe.build_features(df)
    print(f"[Step 2] Feature Engineering Complete: {features_df.shape[1]} total features generated.")
    
    # Step 3: Global Machine Learning Model Training (Validation Split)
    forecaster = GlobalDemandForecaster(use_tweedie=True, tweedie_power=1.1)
    model, val_metrics = forecaster.train_time_series_split(features_df, target_col="sales", val_days=28)
    print(f"[Step 3] Global ML Forecaster Trained. Validation Metrics: {val_metrics}")
    
    # Extract Feature Importance
    importance_df = forecaster.get_feature_importance()
    if not importance_df.empty:
        print("\nTop 5 Most Impactful Features:")
        print(importance_df.head(5).to_string(index=False))
        
    # Generate predictions on the latest 28 days for supply chain optimization
    max_date = features_df["date"].max()
    val_start_date = max_date - pd.Timedelta(days=28)
    val_subset = features_df[features_df["date"] > val_start_date].copy()
    
    val_subset["predicted_sales"] = forecaster.predict(val_subset)
    
    # Step 4: Hierarchical Forecast Reconciliation
    reconciler = HierarchicalReconciler(method="mint_ols")
    reconciled_df = reconciler.reconcile(val_subset, pred_col="predicted_sales")
    print(f"\n[Step 4] Hierarchical Reconciliation Complete across {reconciled_df['dept_id'].nunique()} departments.")
    
    # Step 5: Operations Research Supply Chain Inventory Optimization
    # Summarize predictions per item-store over the 28-day horizon
    forecast_summary = reconciled_df.groupby(["store_id", "item_id"], observed=True).agg(
        pred_mean_demand=("reconciled_sales", "mean"),
        pred_std_demand=("reconciled_sales", "std"),
        sell_price=("sell_price", "mean")
    ).reset_index()
    
    forecast_summary["pred_std_demand"] = forecast_summary["pred_std_demand"].fillna(forecast_summary["pred_mean_demand"] * 0.5)
    
    optimizer = InventoryOptimizer(lead_time_days=7.0, lead_time_std_days=1.5, service_level_z=1.65)
    inventory_df = optimizer.calculate_inventory_policy(forecast_summary)
    
    # Save processed results for Power BI Star-Schema dashboards
    os.makedirs("data/processed", exist_ok=True)
    inventory_df.to_csv("data/processed/Fact_Inventory_Recommender.csv", index=False)
    reconciled_df.to_csv("data/processed/Fact_Daily_Sales_Reconciled.csv", index=False)
    
    # Print Supply Chain Executive Summary
    print("\n=====================================================================================")
    print("SUPPLY CHAIN EXECUTIVE INVENTORY HEALTH SUMMARY")
    print("=====================================================================================")
    status_counts = inventory_df["status"].value_counts()
    for status, count in status_counts.items():
        print(f"  * {status}: {count} SKU-Stores ({count/len(inventory_df)*100:.1f}%)")
        
    total_prevented_loss = inventory_df["prevented_stockout_loss_usd"].sum()
    total_unlocked_capital = inventory_df["overstock_capital_locked_usd"].sum()
    print(f"\n[Financial Impact] Prevented Stockout Revenue Loss: ${total_prevented_loss:,.2f}")
    print(f"[Financial Impact] Unlocked Overstock Working Capital: ${total_unlocked_capital:,.2f}")
    print("=====================================================================================")
    print("Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
