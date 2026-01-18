"""
Streamlit Dashboard for Aadhaar Intelligence Platform

This dashboard provides interactive visualizations and insights for:
- Enrolment patterns and trends
- Update pressure analysis
- Predictive forecasting
- Anomaly detection
- Geospatial analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import load_datasets, preprocess_all_datasets
from utils.indicators import calculate_all_indicators, get_top_districts, create_indicator_summary
# Using Sonnet's robust forecasting module
from utils.forecasting_sonnet import (
    run_forecasting_pipeline, 
    generate_scenario_forecasts,
    train_test_split_temporal,
    compute_mae,
    compute_smape
)
from utils.anomaly import generate_anomaly_report, get_anomaly_summary, get_administrative_suggestions
from utils.geomap import create_india_enrollment_map, create_district_map
from streamlit_folium import st_folium
import folium

# Page configuration
st.set_page_config(
    page_title="Aadhaar Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and preprocess all datasets using Sonnet's robust preprocessing"""
    try:
        # Use Sonnet's preprocessing modules (copied into Opus)
        from utils.preprocessing_sonnet import load_and_preprocess_all_data, merge_all_datasets
        from utils.indicators_sonnet import compute_all_indicators
        
        # Path to the actual data folders (parent of project directory)
        # Resolve to: D:\COURSES\AADHAAR HACKATHON (where api_data folders are)
        DATA_PATH = str(Path(__file__).resolve().parent.parent.parent)
        
        with st.spinner("📊 Loading data from Aadhaar datasets... (this may take a moment)"):
            enrolment, demographic, biometric = load_and_preprocess_all_data(
                DATA_PATH, aggregate_level='state'
            )
            unified_df = merge_all_datasets(enrolment, demographic, biometric, on_columns=['date', 'state'])
            unified_df = compute_all_indicators(unified_df)
        
        # Rename columns to match Opus dashboard expectations (lowercase)
        column_mapping = {
            'Total_Enrolment': 'total_enrolment',
            'Total_Updates': 'total_updates',
            'Total_Demographic_Updates': 'total_demo_updates',
            'Total_Biometric_Updates': 'total_bio_updates',
            'Service_Workload': 'update_pressure_index',
            'Update_Pressure_Index': 'update_pressure_index'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in unified_df.columns and new_col not in unified_df.columns:
                unified_df[new_col] = unified_df[old_col]
        
        # Add district column if not present
        if 'district' not in unified_df.columns:
            unified_df['district'] = 'All Districts'
        
        st.success(f"✅ Loaded {len(unified_df):,} records from Aadhaar datasets")
        return unified_df
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Please ensure data files are available in the parent directory")
        return None


def create_filters(df):
    """Create sidebar filters"""
    st.sidebar.markdown("## 🔍 Filters")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # State filter
    states = ['All'] + sorted(df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)
    
    # District filter (dynamic based on state)
    if selected_state != 'All':
        districts = ['All'] + sorted(df[df['state'] == selected_state]['district'].unique().tolist())
    else:
        districts = ['All'] + sorted(df['district'].unique().tolist())
    
    selected_district = st.sidebar.selectbox("District", districts)
    
    return date_range, selected_state, selected_district


def filter_data(df, date_range, state, district):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_range[0])) &
            (filtered_df['date'] <= pd.Timestamp(date_range[1]))
        ]
    
    # State filter
    if state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == state]
    
    # District filter
    if district != 'All':
        filtered_df = filtered_df[filtered_df['district'] == district]
    
    return filtered_df


def show_overview_metrics(df):
    """Display key metrics in cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_enrolment = df['total_enrolment'].sum()
    total_updates = df['total_updates'].sum()
    avg_upi = df['update_pressure_index'].mean()
    unique_districts = df['district'].nunique()
    
    with col1:
        st.metric("Total Enrolments", f"{total_enrolment:,.0f}")
    
    with col2:
        st.metric("Total Updates", f"{total_updates:,.0f}")
    
    with col3:
        demand_level = "High" if avg_upi > 0.5 else ("Medium" if avg_upi > 0.2 else "Low")
        st.metric("Update Demand Level", demand_level)
    
    with col4:
        st.metric("Districts Covered", f"{unique_districts}")
    
    # Show data usage info
    st.info(f"📊 Analyzing {len(df):,} aggregated records from the complete dataset")


def plot_enrolment_trends(df):
    """Plot enrolment trends over time"""
    st.subheader("📈 Enrolment Trends Over Time")
    
    # Aggregate by date
    daily_enrolment = df.groupby('date')['total_enrolment'].sum().reset_index()
    
    fig = px.line(
        daily_enrolment,
        x='date',
        y='total_enrolment',
        title='Total Enrolments Over Time',
        labels={'total_enrolment': 'Total Enrolments', 'date': 'Date'}
    )
    
    fig.update_traces(line_color='#1f77b4', line_width=2.5)
    fig.update_layout(hovermode='x unified', height=400)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_update_breakdown(df):
    """Plot demographic vs biometric updates"""
    st.subheader("🔄 Update Type Breakdown")
    
    total_demo = df['total_demo_updates'].sum()
    total_bio = df['total_bio_updates'].sum()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Demographic Updates', 'Biometric Updates'],
            values=[total_demo, total_bio],
            hole=0.4,
            marker_colors=['#ff7f0e', '#2ca02c']
        )
    ])
    
    fig.update_layout(title='Demographic vs Biometric Updates', height=400)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_update_pressure_map(df):
    """Plot state-wise update demand heatmap"""
    st.subheader("🗺️ Update Demand by State")
    
    # Aggregate by state
    state_data = df.groupby('state').agg({
        'update_pressure_index': 'mean',
        'total_enrolment': 'sum',
        'total_updates': 'sum'
    }).reset_index()
    
    state_data = state_data.sort_values('update_pressure_index', ascending=False)
    
    fig = px.bar(
        state_data.head(20),
        x='state',
        y='update_pressure_index',
        title='Top 20 States by Update Demand',
        labels={'update_pressure_index': 'Update Demand Ratio', 'state': 'State'},
        color='update_pressure_index',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_top_districts(df):
    """Plot top districts by various metrics"""
    st.subheader("🏆 Top Districts Analysis")
    
    metric_options = {
        'Update Demand Level': 'update_pressure_index',
        'Total Enrolments': 'total_enrolment',
        'Total Updates': 'total_updates'
    }
    
    selected_metric = st.selectbox("Select Metric", list(metric_options.keys()))
    metric_col = metric_options[selected_metric]
    
    top_districts = get_top_districts(df, metric=metric_col, n=15)
    
    if top_districts is not None:
        top_districts['location'] = top_districts['state'] + ' - ' + top_districts['district']
        
        fig = px.bar(
            top_districts,
            x=metric_col,
            y='location',
            orientation='h',
            title=f'Top 15 Districts by {selected_metric}',
            labels={metric_col: selected_metric, 'location': 'District'},
            color=metric_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)


def show_india_map(df):
    """Display interactive India map with enrollment data"""
    st.subheader("🗺️ Interactive India Map - Enrollment Data")
    
    st.info("Click on state markers to view detailed enrollment statistics")
    
    # Select metric to visualize
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metric_display = {
            'total_enrolment': 'Total Enrolments',
            'total_updates': 'Total Updates',
            'update_pressure_index': 'Update Demand Level'
        }
        map_metric = st.selectbox(
            "Select What to Display",
            list(metric_display.keys()),
            format_func=lambda x: metric_display[x]
        )
    
    with col2:
        show_district_map = st.checkbox("Show District Details (for filtered state)")
    
    # If district map is requested, show state selector
    selected_map_state = None
    if show_district_map:
        states = sorted(df['state'].unique().tolist())
        selected_map_state = st.selectbox("Select State for District View", states, key="map_state_selector")
    
    # Create and display map
    try:
        if show_district_map and selected_map_state:
            # Show district-level map
            st.write(f"### District-level Map for {selected_map_state}")
            district_map = create_district_map(df, selected_map_state, map_metric)
            if district_map:
                st_folium(district_map, width=1200, height=600)
            else:
                st.warning(f"No district data available for {selected_map_state}")
        else:
            # Show India-level map
            india_map = create_india_enrollment_map(df, map_metric)
            st_folium(india_map, width=1400, height=700)
        
        # Add summary statistics
        st.markdown("---")
        st.subheader("Map Statistics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        state_agg = df.groupby('state').agg({
            'total_enrolment': 'sum',
            'total_updates': 'sum',
            'update_pressure_index': 'mean'
        })
        
        with col1:
            st.metric("States Covered", len(state_agg))
        
        with col2:
            top_state = state_agg['total_enrolment'].idxmax()
            st.metric("Highest Enrollment", top_state)
        
        with col3:
            high_upi_states = len(state_agg[state_agg['update_pressure_index'] > state_agg['update_pressure_index'].mean()])
            st.metric("High Pressure States", high_upi_states)
        
        with col4:
            avg_enrolment = state_agg['total_enrolment'].mean()
            st.metric("Avg Enrollment/State", f"{avg_enrolment:,.0f}")
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        st.info("Ensure 'folium' and 'streamlit-folium' are installed: pip install folium streamlit-folium")


def plot_enhanced_forecasts(df):
    """Generate and plot forecasts using Sonnet's robust forecasting pipeline"""
    st.subheader("🔮 Trend-Based Forecasting - Future Predictions")
    
    st.info("📊 Using robust trend-focused forecasting that works with limited data. Includes Linear Trend and Holt's methods with scenario projections.")
    st.caption("💡 Works with as few as 6 months of data - perfect for hackathon scenarios!")
    
    # Hierarchical Level Selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_level = st.selectbox(
            "Forecast Level",
            ['National (India)', 'State-wise'],
            help="Choose between national-level or state-specific forecasts"
        )
    
    with col2:
        if forecast_level == 'State-wise':
            states = sorted(df['state'].unique().tolist())
            selected_forecast_state = st.selectbox("Select State for Forecast", states)
        else:
            selected_forecast_state = None
    
    with col3:
        forecast_horizon = st.slider(
            "Months Ahead",
            min_value=3,
            max_value=12,
            value=6,
            help="Number of months to predict"
        )
    
    # Metric Selection
    st.markdown("---")
    st.subheader("📈 What to Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.radio(
            "Forecast Type",
            ['Volume Metrics', 'Update Pressure Index'],
            help="Volume: Predict enrollments/updates | Pressure Index: Predict future system load"
        )
    
    with col2:
        if forecast_type == 'Volume Metrics':
            # Map to Sonnet's column names
            metric_mapping = {
                'Total Enrolment': 'Total_Enrolment',
                'Total Updates': 'Total_Updates',
                'Total Demographic Updates': 'Total_Demographic_Updates',
                'Total Biometric Updates': 'Total_Biometric_Updates'
            }
            selected_metric_display = st.selectbox("Select Metric", list(metric_mapping.keys()))
            forecast_target = metric_mapping[selected_metric_display]
        else:
            forecast_target = 'Service_Workload'
            st.info("🎯 **Policy Impact**: Forecasting workload helps plan resources")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("📊 Analyzing historical data and creating predictions..."):
            try:
                # Filter data based on level
                if forecast_level == 'State-wise' and selected_forecast_state:
                    forecast_data = df[df['state'] == selected_forecast_state].copy()
                    level_label = f"{selected_forecast_state} State"
                else:
                    forecast_data = df.copy()
                    level_label = "National (All India)"
                
                # Aggregate to monthly for forecasting
                forecast_data['date'] = pd.to_datetime(forecast_data['date'])
                
                # Create proper column names expected by Sonnet's forecasting
                if 'total_enrolment' in forecast_data.columns:
                    forecast_data['Total_Enrolment'] = forecast_data['total_enrolment']
                if 'total_updates' in forecast_data.columns:
                    forecast_data['Total_Updates'] = forecast_data['total_updates']
                if 'total_demo_updates' in forecast_data.columns:
                    forecast_data['Total_Demographic_Updates'] = forecast_data['total_demo_updates']
                if 'total_bio_updates' in forecast_data.columns:
                    forecast_data['Total_Biometric_Updates'] = forecast_data['total_bio_updates']
                if 'update_pressure_index' in forecast_data.columns:
                    forecast_data['Service_Workload'] = forecast_data['update_pressure_index']
                
                # Aggregate by month
                monthly_data = forecast_data.groupby(pd.Grouper(key='date', freq='MS')).agg({
                    col: 'sum' for col in forecast_data.select_dtypes(include=[np.number]).columns
                }).reset_index()
                
                # Check if we have enough data
                if len(monthly_data) < 5:
                    st.error(f"❌ Not enough data! Need at least 5 months, got {len(monthly_data)}.")
                    return
                
                st.info(f"🔍 Analyzing {level_label} data with {len(monthly_data)} months of history...")
                
                # Run Sonnet's forecasting pipeline
                results = run_forecasting_pipeline(
                    monthly_data,
                    target_column=forecast_target,
                    date_column='date',
                    n_test_months=min(2, len(monthly_data) // 4),
                    forecast_periods=forecast_horizon
                )
                
                st.caption(f"✓ Using {results['model_name']} | Trend: {results['trend_direction']}")
                
                # Plot results
                fig = go.Figure()
                
                # Historical data (training)
                fig.add_trace(go.Scatter(
                    x=results['train_data'].index,
                    y=results['train_data'].values,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue', width=2)
                ))
                
                # Test data (actual)
                fig.add_trace(go.Scatter(
                    x=results['test_data'].index,
                    y=results['test_data'].values,
                    mode='lines+markers',
                    name='Actual (Test Period)',
                    line=dict(color='green', width=2)
                ))
                
                # Test predictions
                fig.add_trace(go.Scatter(
                    x=results['test_data'].index,
                    y=results['test_predictions'],
                    mode='lines+markers',
                    name='Predicted (Test Period)',
                    line=dict(color='orange', width=2, dash='dot')
                ))
                
                # Future Forecast
                forecast_df = results['future_forecast']
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['forecast'],
                    mode='lines+markers',
                    name='📊 Expected Scenario (Forecast)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                    y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
                
                target_label = forecast_target.replace('_', ' ').title()
                
                fig.update_layout(
                    title=f'{level_label}: {target_label} - {results["model_name"]} Forecast',
                    xaxis_title='Date',
                    yaxis_title=target_label,
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Scenario interpretation
                st.markdown("### 📋 Forecast Scenarios")
                col1, col2, col3 = st.columns(3)
                
                # Show scenarios from Sonnet
                scenarios = results.get('scenarios', {})
                with col1:
                    if 'Conservative' in scenarios:
                        cons_val = scenarios['Conservative']['forecast'].mean()
                        st.info(f"**📉 Conservative**\n{cons_val:,.0f}")
                with col2:
                    if 'Expected' in scenarios:
                        exp_val = scenarios['Expected']['forecast'].mean()
                        st.success(f"**📊 Expected**\n{exp_val:,.0f}")
                with col3:
                    if 'Optimistic' in scenarios:
                        opt_val = scenarios['Optimistic']['forecast'].mean()
                        st.warning(f"**📈 Optimistic**\n{opt_val:,.0f}")
                
                # Show forecast table
                st.subheader("📅 Predicted Values")
                st.dataframe(
                    forecast_df[['date', 'forecast', 'lower_ci', 'upper_ci']].rename(columns={
                        'date': 'Date',
                        'forecast': 'Expected',
                        'lower_ci': 'Lower Bound',
                        'upper_ci': 'Upper Bound'
                    }).style.format({'Expected': '{:,.0f}', 'Lower Bound': '{:,.0f}', 'Upper Bound': '{:,.0f}'}),
                    use_container_width=True
                )
                
                # Accuracy metrics
                st.markdown("---")
                st.subheader("📊 Model Accuracy Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mae_val = results['evaluation']['mae']
                    st.metric("MAE", f"{mae_val:,.0f}", help="Mean Absolute Error")
                
                with col2:
                    smape = results['evaluation']['smape']
                    accuracy = 100 - smape
                    st.metric("Accuracy", f"{accuracy:.1f}%", help="Based on sMAPE")
                
                with col3:
                    st.metric("Trend", results['trend_direction'], help="Overall trend direction")
                
                # Interpretation
                st.markdown("### 📖 Forecast Interpretation")
                st.info(results['methodology'])
                
                st.success("✅ Predictions generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Try selecting a different metric or forecast level.")


def show_anomalies(df):
    """Detect and display operational signals (anomalies)"""
    st.subheader("⚠️ Operational Load Signals - Anomaly Detection")
    
    st.info("📍 Detecting locations with unusual activity that may indicate operational load spikes or data quality issues")
    
    # Z-score based anomaly detection
    from scipy import stats
    
    metric_names = {
        'update_pressure_index': 'Update Demand Level',
        'total_updates': 'Total Updates',
        'total_enrolment': 'Total Enrolments'
    }
    metric = st.selectbox("What to Check for Unusual Activity", 
                         list(metric_names.keys()),
                         format_func=lambda x: metric_names[x])
    
    # Calculate statistical outliers
    df_copy = df.copy()
    df_copy['z_score'] = np.abs(stats.zscore(df_copy[metric].fillna(0)))
    
    # Flag anomalies (statistical threshold)
    anomalies = df_copy[df_copy['z_score'] > 3]
    
    st.write(f"**🔍 Found {len(anomalies)} operational signals (unusual activity patterns)**")
    st.caption("These locations show significantly different patterns from average - may indicate load spikes or data issues")
    
    if len(anomalies) > 0:
        # Analyze duration and repetition
        anomaly_duration = anomalies.groupby(['state', 'district']).size().reset_index(name='days_flagged')
        st.markdown(f"**Duration Analysis**: {len(anomaly_duration[anomaly_duration['days_flagged'] > 1])} locations show repeated anomalies (multi-day)")
        
        # Show anomalies by state
        anomaly_by_state = anomalies.groupby('state').size().reset_index(name='count')
        anomaly_by_state = anomaly_by_state.sort_values('count', ascending=False)
        
        fig = px.bar(
            anomaly_by_state.head(15),
            x='state',
            y='count',
            title='Operational Load Signals by State',
            labels={'count': 'Number of Signals', 'state': 'State'},
            color='count',
            color_continuous_scale='Oranges'
        )
        
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        st.subheader("📋 Operational Signal Details (Top 20)")
        st.caption("🔴 High priority: Repeated signals | 🟡 Medium: Single occurrence")
        
        anomaly_display = anomalies.nlargest(20, 'z_score')[['date', 'state', 'district', metric]].copy()
        anomaly_display = anomaly_display.rename(columns={metric: metric_names[metric]})
        
        # Add priority flag based on repetition
        anomaly_display['Status'] = anomaly_display.apply(
            lambda row: "🔴 Repeated" if len(anomalies[(anomalies['state'] == row['state']) & (anomalies['district'] == row['district'])]) > 1 else "🟡 Single",
            axis=1
        )
        
        st.dataframe(anomaly_display, use_container_width=True)
    else:
        st.success("✅ No significant anomalies detected - operations appear normal")


def show_insights(df):
    """Generate and display key insights"""
    st.subheader("💡 Key Insights & Recommendations")
    
    # Policy Recommendations Panel
    st.markdown("### 🎯 What Should UIDAI Do? - Actionable Recommendations")
    
    # Calculate key statistics for recommendations
    top_state = df.groupby('state')['total_enrolment'].sum().idxmax()
    top_state_enrolment = df.groupby('state')['total_enrolment'].sum().max()
    
    avg_upi = df['update_pressure_index'].mean()
    high_upi_districts = len(df[df['update_pressure_index'] > avg_upi * 1.5])
    
    demo_total = df['total_demo_updates'].sum()
    bio_total = df['total_bio_updates'].sum()
    
    # Create recommendations table
    recommendations_data = []
    
    if avg_upi > 0.3:
        recommendations_data.append({
            "Insight": "High Update Pressure Detected",
            "Suggested Action": "Deploy temporary staffing in high-demand districts",
            "Priority": "🔴 High"
        })
    
    if high_upi_districts > 100:
        recommendations_data.append({
            "Insight": f"{high_upi_districts} districts show above-average load",
            "Suggested Action": "Resource reallocation from low-demand areas",
            "Priority": "🟡 Medium"
        })
    
    # Check for monthly patterns
    monthly_avg = df.groupby(df['date'].dt.month)['total_enrolment'].mean()
    if monthly_avg.max() / monthly_avg.min() > 1.5:
        peak_month = monthly_avg.idxmax()
        recommendations_data.append({
            "Insight": f"Seasonal spike observed in month {peak_month}",
            "Suggested Action": "Pre-deployment of resources 2-3 weeks before peak",
            "Priority": "🟡 Medium"
        })
    
    # Demo vs Bio ratio
    if demo_total / bio_total > 1.3:
        recommendations_data.append({
            "Insight": "Higher demographic update demand",
            "Suggested Action": "Balance workload between update centers",
            "Priority": "🟢 Low"
        })
    
    # Always add forecast-based planning
    recommendations_data.append({
        "Insight": "12-month forecast available",
        "Suggested Action": "Use SARIMA predictions for 6-12 month capacity planning",
        "Priority": "🔴 High"
    })
    
    if recommendations_data:
        rec_df = pd.DataFrame(recommendations_data)
        st.table(rec_df)
    
    st.markdown("---")
    
    # Calculate key statistics
    top_state = df.groupby('state')['total_enrolment'].sum().idxmax()
    top_state_enrolment = df.groupby('state')['total_enrolment'].sum().max()
    
    avg_upi = df['update_pressure_index'].mean()
    high_upi_districts = len(df[df['update_pressure_index'] > avg_upi * 1.5])
    
    demo_total = df['total_demo_updates'].sum()
    bio_total = df['total_bio_updates'].sum()
    
    # Display insights
    demand_level_text = "high" if avg_upi > 0.5 else ("moderate" if avg_upi > 0.2 else "low")
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>📊 Enrolment Overview</h4>
    <ul>
        <li><strong>{top_state}</strong> has the highest total enrolments with <strong>{top_state_enrolment:,.0f}</strong> registrations</li>
        <li>Overall update demand level is <strong>{demand_level_text}</strong> across all regions</li>
        <li><strong>{high_upi_districts}</strong> districts have higher than average update demands</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    ratio = demo_total/bio_total if bio_total > 0 else 0
    ratio_text = "more demographic" if ratio > 1.2 else ("more biometric" if ratio < 0.8 else "balanced")
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>🔄 Update Activity Summary</h4>
    <ul>
        <li>Personal Information Updates: <strong>{demo_total:,.0f}</strong></li>
        <li>Biometric Updates: <strong>{bio_total:,.0f}</strong></li>
        <li>Update pattern is <strong>{ratio_text}</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Suggested Actions</h4>
    <ul>
        <li>Allocate more staff and resources to districts with high update demand</li>
        <li>Investigate locations showing unusual activity patterns</li>
        <li>Use predictions to plan for future capacity needs 6-12 months ahead</li>
        <li>Check states with decreasing enrollment trends to identify potential issues</li>
        <li>Balance workload between demographic and biometric update centers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Quality & Limitations
    st.markdown("---")
    st.markdown("### 📋 Data Quality & Analytical Limitations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Data Characteristics:**
        - Aggregated state/district level data
        - Monthly time-series patterns
        - 12 months of historical data (2023)
        - 50,000+ aggregated records
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Important Limitations:**
        - No individual-level data (privacy preserved)
        - No causal inference possible
        - Forecasts are planning-level estimates
        - Anomalies require manual investigation
        """)
    
    st.info("💡 **Note**: This platform provides decision support for administrative planning.  All recommendations should be validated with domain experts before implementation.")


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🏛️ Aadhaar Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("### Data-Driven Analytics & Decision Support System")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    date_range, selected_state, selected_district = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, date_range, selected_state, selected_district)
    
    # Show filtered data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Filtered Data Info")
    st.sidebar.write(f"Records: {len(filtered_df):,}")
    st.sidebar.write(f"States: {filtered_df['state'].nunique()}")
    st.sidebar.write(f"Districts: {filtered_df['district'].nunique()}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🗺️ India Map",
        "🔮 Forecasting",
        "⚠️ Anomaly Detection",
        "💡 Insights"
    ])
    
    with tab1:
        show_overview_metrics(filtered_df)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            plot_enrolment_trends(filtered_df)
        with col2:
            plot_update_breakdown(filtered_df)
    
    with tab2:
        show_india_map(filtered_df)
    
    with tab3:
        plot_enhanced_forecasts(filtered_df)
    
    with tab4:
        show_anomalies(filtered_df)
    
    with tab5:
        show_insights(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>⚠️ Privacy Notice:</strong> All data is aggregated and anonymised. 
        No individual-level inference is attempted. Outputs are advisory only.</p>
        <p>Aadhaar Intelligence Platform | Built for Governance & Service Delivery</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
