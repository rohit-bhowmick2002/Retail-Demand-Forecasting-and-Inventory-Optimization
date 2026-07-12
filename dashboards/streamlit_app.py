"""
Interactive Streamlit Dashboard: AI-Powered Retail Demand Forecasting & Inventory Optimization
To run locally: streamlit run dashboards/streamlit_app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Retail Inventory Command Center | AI & Operations Research",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛒 AI-Powered Retail Demand Forecasting & Inventory Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Combining Global Machine Learning (LightGBM Tweedie) with Mathematical Operations Research (Safety Stock, ROP & EOQ)</p>', unsafe_allow_html=True)

# Sidebar - Operations Research Simulation Controls
st.sidebar.header("⚙️ Supply Chain Policy Simulator")
st.sidebar.markdown("Adjust mathematical parameters below to see how safety buffers and reorder quantities shift across the network.")

lead_time = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1.0, max_value=21.0, value=7.0, step=0.5)
lead_time_std = st.sidebar.slider("Lead Time Volatility (Std Dev Days)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
service_level_pct = st.sidebar.selectbox("Target Service Level (Z-Score)", options=[("90% (Z=1.28)", 1.28), ("95% (Z=1.65)", 1.65), ("98% (Z=2.05)", 2.05), ("99% (Z=2.33)", 2.33)], index=1)
z_score = service_level_pct[1]
holding_rate = st.sidebar.slider("Annual Holding Cost Rate (%)", min_value=10, max_value=50, value=25, step=5) / 100.0
order_cost = st.sidebar.number_input("Fixed Order Cost ($ per PO)", min_value=5.0, max_value=100.0, value=15.0, step=5.0)

# Load data or generate synthetic on the fly
@st.cache_data
def load_sample_data():
    sample_path = os.path.join(os.path.dirname(__file__), "../data/processed/Fact_Inventory_Recommender.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Fallback synthetic generation for Streamlit cloud demo
        np.random.seed(42)
        skus = [f"FOODS_3_{i:03d}" for i in range(50)] + [f"HOBBIES_1_{i:03d}" for i in range(30)] + [f"HOUSEHOLD_1_{i:03d}" for i in range(40)]
        stores = ["CA_1", "CA_2", "TX_1", "WI_1"]
        data = []
        for sku in skus:
            for store in stores:
                mean_d = np.random.uniform(1.2, 12.0)
                data.append({
                    "store_id": store,
                    "item_id": sku,
                    "dept_id": sku.split("_")[0] + "_" + sku.split("_")[1],
                    "sell_price": round(np.random.uniform(2.49, 18.99), 2),
                    "pred_mean_demand": round(mean_d, 2),
                    "pred_std_demand": round(mean_d * np.random.uniform(0.4, 1.1), 2),
                    "current_stock": int(mean_d * np.random.uniform(3, 25))
                })
        return pd.DataFrame(data)

df = load_sample_data()

# Recalculate OR Policies dynamically based on Sidebar sliders
df["lead_time_demand"] = df["pred_mean_demand"] * lead_time
variance_term = (lead_time * (df["pred_std_demand"] ** 2)) + ((df["pred_mean_demand"] ** 2) * (lead_time_std ** 2))
df["safety_stock"] = np.ceil(z_score * np.sqrt(np.maximum(0.0, variance_term)))
df["reorder_point"] = np.ceil(df["lead_time_demand"] + df["safety_stock"])

annual_demand = df["pred_mean_demand"] * 365.0
unit_holding = np.where(df["sell_price"] * holding_rate > 0, df["sell_price"] * holding_rate, 0.50)
df["eoq_reorder_qty"] = np.ceil(np.sqrt((2.0 * annual_demand * order_cost) / unit_holding))

# Status classification
crit_mask = df["current_stock"] <= df["reorder_point"]
over_mask = df["current_stock"] > (df["reorder_point"] + (2.5 * df["eoq_reorder_qty"]))

df["status"] = "OPTIMAL: Balanced Inventory"
df.loc[crit_mask, "status"] = "CRITICAL: Stockout Risk (Order Now)"
df.loc[over_mask, "status"] = "WARNING: Overstocked (Discount/Stop Order)"

# Financial impact
df["prevented_stockout_loss_usd"] = np.where(crit_mask, df["pred_mean_demand"] * 30.0 * df["sell_price"] * 0.95, 0.0)
df["overstock_capital_locked_usd"] = np.where(over_mask, (df["current_stock"] - (df["reorder_point"] + df["eoq_reorder_qty"])) * df["sell_price"], 0.0)

# Top Executive KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Total SKUs Monitored", value=f"{len(df):,}", delta="Multi-Store Network")
with kpi2:
    crit_count = crit_mask.sum()
    st.metric(label="Critical Stockout Alerts", value=f"{crit_count:,}", delta=f"{-crit_count/len(df)*100:.1f}% of Catalog", delta_color="inverse")
with kpi3:
    prevented_loss = df["prevented_stockout_loss_usd"].sum()
    st.metric(label="30-Day Prevented Lost Revenue", value=f"${prevented_loss:,.2f}", delta="▲ High Priority POs")
with kpi4:
    overstock_locked = df["overstock_capital_locked_usd"].sum()
    st.metric(label="Overstock Capital Unlocked", value=f"${overstock_locked:,.2f}", delta="▼ Holding Cost Cut")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["🚨 Critical Procurement Alerts", "📊 Inventory Health by Department", "🔍 SKU Deep-Dive & Math Inspector"])

with tab1:
    st.subheader("Automated Purchase Order Generator (Items Below Reorder Point)")
    crit_df = df[df["status"].str.startswith("CRITICAL")].sort_values(by="prevented_stockout_loss_usd", ascending=False)
    if crit_df.empty:
        st.success("🎉 All monitored SKUs are currently above their Reorder Points! No emergency POs needed.")
    else:
        st.dataframe(
            crit_df[["store_id", "item_id", "current_stock", "reorder_point", "eoq_reorder_qty", "prevented_stockout_loss_usd", "status"]]
            .rename(columns={
                "store_id": "Store",
                "item_id": "SKU Item ID",
                "current_stock": "Current Stock",
                "reorder_point": "Reorder Point (ROP)",
                "eoq_reorder_qty": "Recommended EOQ Order",
                "prevented_stockout_loss_usd": "Potential Lost Revenue ($)"
            }),
            use_container_width=True
        )

with tab2:
    st.subheader("Inventory Status Distribution Across Departments")
    dept_summary = pd.crosstab(df["dept_id"], df["status"])
    st.bar_chart(dept_summary)

with tab3:
    st.subheader("Single SKU Operations Research Math Inspector")
    selected_sku = st.selectbox("Select SKU Item ID to Inspect:", options=df["item_id"].unique())
    sku_data = df[df["item_id"] == selected_sku].iloc[0]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**SKU ID:** `{sku_data['item_id']}` | **Store:** `{sku_data['store_id']}` | **Unit Price:** `${sku_data['sell_price']}`")
        st.markdown(f"- **Forecasted Mean Daily Demand ($\mu_d$):** `{sku_data['pred_mean_demand']} units/day`")
        st.markdown(f"- **Forecasted Demand Volatility ($\sigma_d$):** `{sku_data['pred_std_demand']} units/day`")
        st.markdown(f"- **Current Warehouse Stock:** `{int(sku_data['current_stock'])} units`")
    with c2:
        st.markdown(f"#### 📐 Calculated OR Policy Matrix")
        st.markdown(f"- **Safety Stock ($SS$):** `{int(sku_data['safety_stock'])} units` *(Z={z_score})*")
        st.markdown(f"- **Reorder Point ($ROP$):** `{int(sku_data['reorder_point'])} units` *(Lead Time Demand + SS)*")
        st.markdown(f"- **Economic Order Quantity ($EOQ$):** `{int(sku_data['eoq_reorder_qty'])} units`")
        st.info(f"**Current Status:** {sku_data['status']}")
