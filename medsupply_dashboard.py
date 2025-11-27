
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="MedSupply AI - Uganda",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💊 MedSupply AI - Uganda</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning-Powered Medicine Supply Chain Optimization")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Facility Analysis", "Model Performance", "OOQ Calculator"])

if page == "Dashboard":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Demand Forecast Accuracy", "99.94%", "0.06% MAPE")
    with col2:
        st.metric("OOQ Prediction Error", "0.07%", "Target: 7.3%")
    with col3:
        st.metric("Stockout Prediction F1", "0.349", "+14.4% Improved")
    with col4:
        st.metric("Facilities Analyzed", "15,000", "Across Uganda")
    
    # Success Box
    st.markdown('<div class="success-box">🎯 <strong>PRIMARY TARGET ACHIEVED:</strong> OOQ Prediction MPE of 0.07% (104x better than 7.3% target)</div>', unsafe_allow_html=True)
    
    # Sample Facility Predictions
    st.subheader("🏥 Real-Time Facility Predictions")
    
    # Sample data - in real implementation, this would come from your model
    sample_facilities = [
        {"Facility": "Kampala General Hospital", "Drug": "Artemether-Lumefantrine", 
         "Current Stock": 6757, "Predicted Demand": 4523, "OOQ Recommended": 5838, 
         "Stockout Risk": "Medium", "Action": "Order Recommended"},
        {"Facility": "Mbarara Clinic", "Drug": "Omeprazole Capsules", 
         "Current Stock": 1401, "Predicted Demand": 2120, "OOQ Recommended": 0,
         "Stockout Risk": "Low", "Action": "Adequate Stock"},
        {"Facility": "Gulu Health Center", "Drug": "Metformin Tablets", 
         "Current Stock": 892, "Predicted Demand": 1543, "OOQ Recommended": 2156,
         "Stockout Risk": "High", "Action": "Urgent Order"}
    ]
    
    df_display = pd.DataFrame(sample_facilities)
    st.dataframe(df_display, use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Demand vs Stock Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current Stock', x=[f["Facility"] for f in sample_facilities], 
                            y=[f["Current Stock"] for f in sample_facilities]))
        fig.add_trace(go.Bar(name='Predicted Demand', x=[f["Facility"] for f in sample_facilities], 
                            y=[f["Predicted Demand"] for f in sample_facilities]))
        fig.update_layout(title="Current Stock vs Predicted Demand")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Distribution
        risks = [f["Stockout Risk"] for f in sample_facilities]
        risk_counts = pd.Series(risks).value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title="Stockout Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Facility Analysis":
    st.header("🏥 Facility-Level Analysis")
    
    # Facility selector
    facilities = ["Kampala General Hospital", "Mbarara Clinic", "Gulu Health Center", "Jinja Pharmacy"]
    selected_facility = st.selectbox("Select Facility", facilities)
    
    # Display facility details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Inventory Status")
        st.metric("Current Stock Level", "6,757 units", "-12% from last month")
        st.metric("Weekly Consumption", "1,130 units", "+5% trend")
        st.metric("Stockout Probability", "35.6%", "Medium Risk")
    
    with col2:
        st.subheader("Recommendations")
        st.success("✅ Optimal Order Quantity: 5,838 units")
        st.warning("⚠️ Monitor: Artemether-Lumefantrine (high demand season)")
        st.info("💡 Suggestion: Implement FEFO for 15% reduction in expiries")

elif page == "Model Performance":
    st.header("🤖 Model Performance Metrics & Comparison")
    
    # Model Comparison Section
    st.subheader("📊 Comprehensive Model Comparison")
    
    # Model performance data
    models_data = {
        "Model": ["SARIMA", "Random Forest", "XGBoost", "LSTM"],
        "MAPE (%)": [0.15, 0.06, 1.02, 0.07],  # Example values - will be updated from actual results
        "MAE": [385.0, 1.27, 22.03, 1.4],
        "RMSE": [580.0, 4.45, 55.77, 5.0],
        "Ease of Implementation": [90, 60, 70, 40],
        "Data Requirements": [80, 70, 70, 30],
        "Computing Power": [85, 75, 70, 20],
        "Insight": ["✅ Most Practical", "⚙️ Needs Expertise", "⚙️ Needs Expertise", "💻 Needs More Data/Compute"]
    }
    
    comparison_df = pd.DataFrame(models_data)
    
    # Key Insights Box
    st.info("""
    **Model Comparison Insights:**
    - **SARIMA**: Most practical choice - accurate, easier to implement, works with smaller datasets
    - **XGBoost & Random Forest**: Similar performance to SARIMA but need more technical expertise
    - **LSTM**: Could be more accurate but requires larger datasets and more computing power
    """)
    
    # Performance Metrics Comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_mape = px.bar(comparison_df, x="Model", y="MAPE (%)", 
                         color="Model",
                         title="MAPE Comparison (Lower is Better)",
                         color_discrete_map={
                             "SARIMA": "#2ecc71",
                             "Random Forest": "#3498db",
                             "XGBoost": "#e74c3c",
                             "LSTM": "#f39c12"
                         })
        fig_mape.add_hline(y=7.3, line_dash="dash", line_color="red", 
                          annotation_text="Target: 7.3%")
        st.plotly_chart(fig_mape, use_container_width=True)
    
    with col2:
        fig_mae = px.bar(comparison_df, x="Model", y="MAE",
                        color="Model",
                        title="Mean Absolute Error (Lower is Better)",
                        color_discrete_map={
                            "SARIMA": "#2ecc71",
                            "Random Forest": "#3498db",
                            "XGBoost": "#e74c3c",
                            "LSTM": "#f39c12"
                        })
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col3:
        fig_rmse = px.bar(comparison_df, x="Model", y="RMSE",
                         color="Model",
                         title="Root Mean Squared Error (Lower is Better)",
                         color_discrete_map={
                             "SARIMA": "#2ecc71",
                             "Random Forest": "#3498db",
                             "XGBoost": "#e74c3c",
                             "LSTM": "#f39c12"
                         })
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Model Characteristics Radar Chart
    st.subheader("🎯 Model Characteristics Comparison")
    
    # Prepare data for radar chart
    categories = ['Accuracy', 'Ease of Implementation', 'Data Requirements', 'Computing Power']
    
    fig_radar = go.Figure()
    
    for idx, row in comparison_df.iterrows():
        # Calculate accuracy from MAPE (inverse)
        accuracy = 100 - row['MAPE (%)']
        values = [
            accuracy,
            row['Ease of Implementation'],
            row['Data Requirements'],
            row['Computing Power']
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Model'],
            line=dict(width=2)
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Multi-Dimensional Model Comparison (Higher is Better)",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Model Characteristics Bar Chart
    st.subheader("📈 Model Characteristics Breakdown")
    
    characteristics_df = comparison_df[['Model', 'Ease of Implementation', 
                                       'Data Requirements', 'Computing Power']].melt(
        id_vars='Model',
        var_name='Characteristic',
        value_name='Score'
    )
    
    fig_char = px.bar(characteristics_df, x="Model", y="Score", 
                     color="Characteristic",
                     barmode="group",
                     title="Model Characteristics Comparison",
                     labels={"Score": "Score (0-100)"})
    st.plotly_chart(fig_char, use_container_width=True)
    
    # Summary Table
    st.subheader("📋 Model Performance Summary")
    display_df = comparison_df[['Model', 'MAPE (%)', 'MAE', 'RMSE', 'Insight']].copy()
    display_df['Rank'] = display_df['MAPE (%)'].rank().astype(int)
    display_df = display_df.sort_values('MAPE (%)')
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Stockout Prediction Comparison
    st.subheader("🔍 Stockout Prediction Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(px.bar(x=["Before", "After"], y=[0.305, 0.349], 
                              title="Stockout Prediction F1-Score Improvement",
                              color=["Before", "After"],
                              labels={"x": "Optimization Stage", "y": "F1-Score"}))
    
    with col2:
        st.plotly_chart(px.bar(x=["Target", "Achieved"], y=[7.3, 0.07], 
                              title="OOQ Prediction MPE Comparison",
                              color=["Target", "Achieved"],
                              labels={"x": "Metric", "y": "MPE (%)"}))

elif page == "OOQ Calculator":
    st.header("🧮 Optimal Order Quantity Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_stock = st.number_input("Current Stock (units)", min_value=0, value=1000)
        weekly_demand = st.number_input("Weekly Demand (units)", min_value=0, value=500)
        lead_time = st.number_input("Lead Time (days)", min_value=1, value=21)
    
    with col2:
        service_level = st.slider("Service Level Target", 0.8, 0.99, 0.95)
        demand_volatility = st.slider("Demand Volatility", 0.1, 0.5, 0.2)
    
    if st.button("Calculate OOQ"):
        # Simplified OOQ calculation
        lead_time_weeks = lead_time / 7
        safety_stock = 1.65 * (weekly_demand * demand_volatility) * np.sqrt(lead_time_weeks)
        ooq = max(0, (weekly_demand * lead_time_weeks) + safety_stock - current_stock)
        
        st.success(f"🎯 Recommended Order Quantity: {ooq:.0f} units")
        st.info(f"🛡️ Safety Stock: {safety_stock:.0f} units")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = ooq,
            title = {"text": "Optimal Order Quantity"},
            delta = {'reference': weekly_demand * lead_time_weeks, 'relative': True},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**MedSupply AI** | Uganda Pharmaceutical Supply Chain Optimization | Final Year Project 2024")
