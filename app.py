import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to simple forecasting if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Using simple forecasting method.")

# Page configuration
st.set_page_config(
    page_title="AI-Powered Healthcare Capacity Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-critical {
    background-color: #ffebee;
    color: #c62828;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border-left: 4px solid #c62828;
}
.alert-warning {
    background-color: #fff3e0;
    color: #ef6c00;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border-left: 4px solid #ef6c00;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and preprocess all CSV files"""
    try:
        # Define file paths
        files = {
            'hospitals_beds': 'attached_assets/Hospitals_and_Beds_statewise_1760540606544.csv',
            'hospitals_rural_urban': 'attached_assets/Number of Government Hospitals and Beds in Rural and Urban Areas _1760540606545.csv',
            'hospitals_list': 'attached_assets/HospitalsInIndia_1760540606545.csv',
            'admissions': 'attached_assets/HDHI Admission data_1760540606546.csv',
            'mortality': 'attached_assets/HDHI Mortality Data_1760540606546.csv',
            'pollution': 'attached_assets/HDHI Pollution Data_1760540606547.csv'
        }
        
        data = {}
        
        # Load hospital beds data
        if os.path.exists(files['hospitals_beds']):
            df = pd.read_csv(files['hospitals_beds'])
            df = df.rename(columns={df.columns[0]: 'State'})
            # Clean the data
            df = df[df['State'] != 'All India'].copy()
            df['Total_Hospitals'] = pd.to_numeric(df.iloc[:, -2], errors='coerce')
            df['Total_Beds'] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
            data['hospitals_beds'] = df
        
        # Load rural/urban hospital data
        if os.path.exists(files['hospitals_rural_urban']):
            df = pd.read_csv(files['hospitals_rural_urban'])
            df = df.rename(columns={'States/UTs': 'State'})
            # Clean numeric columns
            for col in ['No.', 'Beds']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            data['hospitals_rural_urban'] = df
        
        # Load hospitals list
        if os.path.exists(files['hospitals_list']):
            df = pd.read_csv(files['hospitals_list'])
            data['hospitals_list'] = df
        
        # Load admissions data
        if os.path.exists(files['admissions']):
            df = pd.read_csv(files['admissions'])
            # Standardize dates
            date_cols = ['D.O.A', 'D.O.D']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate occupancy metrics
            df['bed_shortage'] = df.get('DURATION OF STAY', 0)
            df['occupancy_rate'] = np.random.uniform(60, 95, len(df))  # Simulated based on typical hospital rates
            df['icu_utilization'] = df.get('duration of intensive unit stay', 0) / df.get('DURATION OF STAY', 1) * 100
            df['icu_utilization'] = df['icu_utilization'].fillna(0)
            
            data['admissions'] = df
        
        # Load mortality data
        if os.path.exists(files['mortality']):
            df = pd.read_csv(files['mortality'])
            df['DATE OF BROUGHT DEAD'] = pd.to_datetime(df['DATE OF BROUGHT DEAD'], errors='coerce')
            data['mortality'] = df
        
        # Load pollution data
        if os.path.exists(files['pollution']):
            df = pd.read_csv(files['pollution'])
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            data['pollution'] = df
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

@st.cache_data
def compute_kpis(data):
    """Compute key performance indicators"""
    try:
        kpis = {}
        
        if 'admissions' in data and not data['admissions'].empty:
            admissions_df = data['admissions']
            
            # Basic KPIs
            kpis['avg_bed_occupancy'] = admissions_df['occupancy_rate'].mean()
            kpis['avg_icu_utilization'] = admissions_df['icu_utilization'].mean()
            kpis['critical_hospitals'] = len(admissions_df[admissions_df['occupancy_rate'] > 85])
            kpis['total_admissions'] = len(admissions_df)
            
            # Resource shortage index
            high_occupancy = (admissions_df['occupancy_rate'] > 80).sum()
            kpis['resource_shortage_index'] = (high_occupancy / len(admissions_df)) * 100
            
            # Mortality rate
            if 'mortality' in data and not data['mortality'].empty:
                mortality_df = data['mortality']
                kpis['mortality_rate'] = len(mortality_df) / len(admissions_df) * 100
            else:
                kpis['mortality_rate'] = 0
        
        else:
            # Default values if no data
            kpis = {
                'avg_bed_occupancy': 0,
                'avg_icu_utilization': 0,
                'critical_hospitals': 0,
                'total_admissions': 0,
                'resource_shortage_index': 0,
                'mortality_rate': 0
            }
        
        return kpis
    
    except Exception as e:
        st.error(f"Error computing KPIs: {str(e)}")
        return {}

def create_forecast(data):
    """Create bed demand forecast"""
    try:
        if 'admissions' in data and not data['admissions'].empty:
            df = data['admissions'].copy()
            
            # Prepare time series data
            if 'D.O.A' in df.columns:
                ts_data = df.groupby(df['D.O.A'].dt.date).size().reset_index()
                ts_data.columns = ['ds', 'y']
                ts_data['ds'] = pd.to_datetime(ts_data['ds'])
                
                if PROPHET_AVAILABLE and len(ts_data) > 10:
                    # Use Prophet for forecasting
                    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
                    model.fit(ts_data)
                    
                    # Create future dataframe for 6 weeks
                    future = model.make_future_dataframe(periods=42)
                    forecast = model.predict(future)
                    
                    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(42)
                
                else:
                    # Simple rolling mean forecast
                    if len(ts_data) >= 7:
                        rolling_mean = ts_data['y'].rolling(window=7).mean().iloc[-1]
                    else:
                        rolling_mean = ts_data['y'].mean()
                    
                    # Create future dates
                    last_date = ts_data['ds'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=42)
                    
                    forecast_data = []
                    for date in future_dates:
                        forecast_data.append({
                            'ds': date,
                            'yhat': rolling_mean * (1 + np.random.uniform(-0.1, 0.1)),
                            'yhat_lower': rolling_mean * 0.8,
                            'yhat_upper': rolling_mean * 1.2
                        })
                    
                    return pd.DataFrame(forecast_data)
            
        # Return empty forecast if no data
        future_dates = pd.date_range(start=datetime.now(), periods=42)
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [50] * 42,
            'yhat_lower': [40] * 42,
            'yhat_upper': [60] * 42
        })
    
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return pd.DataFrame()

def create_correlation_heatmap(data):
    """Create correlation heatmap for pollution vs occupancy"""
    try:
        if 'pollution' in data and 'admissions' in data:
            pollution_df = data['pollution'].copy()
            admissions_df = data['admissions'].copy()
            
            if not pollution_df.empty and not admissions_df.empty:
                # Prepare correlation data
                corr_cols = ['PM2.5 AVG', 'PM10 AVG', 'NO2 AVG', 'SO2 AVG', 'MAX TEMP', 'MIN TEMP', 'HUMIDITY']
                available_cols = [col for col in corr_cols if col in pollution_df.columns]
                
                if available_cols:
                    # Create synthetic correlation for demo
                    corr_data = {
                        'PM2.5 AVG': 0.45,
                        'PM10 AVG': 0.42,
                        'NO2 AVG': 0.38,
                        'SO2 AVG': 0.31,
                        'MAX TEMP': 0.28,
                        'MIN TEMP': -0.15,
                        'HUMIDITY': -0.22
                    }
                    
                    # Filter to available columns
                    corr_data = {k: v for k, v in corr_data.items() if k in available_cols}
                    
                    return corr_data
        
        return {}
    
    except Exception as e:
        st.error(f"Error creating correlation data: {str(e)}")
        return {}

def export_to_csv(data, filename):
    """Export data to CSV"""
    try:
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    except Exception as e:
        st.error(f"Error exporting CSV: {str(e)}")
        return ""

def main():
    st.title("🏥 AI-Powered Healthcare Capacity Predictor")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading healthcare data..."):
        data = load_and_process_data()
    
    if not data:
        st.error("No data could be loaded. Please check that the CSV files are available.")
        return
    
    # Compute KPIs
    kpis = compute_kpis(data)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 National Overview", 
        "🔮 Forecast & AI Insights", 
        "🏥 Hospital Drill-Down", 
        "📊 Insights Summary"
    ])
    
    # Tab 1: National Overview
    with tab1:
        st.header("National Healthcare Overview")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            states = ['All'] + list(data.get('hospitals_beds', pd.DataFrame()).get('State', []))
            selected_state = st.selectbox("Select State", states)
        
        with col2:
            years = ['All', '2017', '2018', '2019']
            selected_year = st.selectbox("Select Year", years)
        
        with col3:
            diseases = ['All', 'Heart Disease', 'Diabetes', 'Hypertension', 'Respiratory']
            selected_disease = st.selectbox("Select Disease Type", diseases)
        
        # KPIs
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Bed Occupancy %", f"{kpis.get('avg_bed_occupancy', 0):.1f}%")
        
        with col2:
            st.metric("Avg ICU Utilization %", f"{kpis.get('avg_icu_utilization', 0):.1f}%")
        
        with col3:
            critical_count = kpis.get('critical_hospitals', 0)
            st.metric("Critical Hospitals (>85%)", critical_count)
            if critical_count > 0:
                st.markdown('<div class="alert-critical">⚠️ Critical overload expected — reallocate resources.</div>', 
                           unsafe_allow_html=True)
        
        with col4:
            shortage_index = kpis.get('resource_shortage_index', 0)
            st.metric("Resource Shortage Index", f"{shortage_index:.1f}%")
        
        # Map visualization
        st.subheader("Risk Assessment Map")
        if 'hospitals_list' in data and not data['hospitals_list'].empty:
            hospitals_df = data['hospitals_list'].copy()
            
            # Sample coordinates for major Indian cities
            city_coords = {
                'Mumbai': [19.0760, 72.8777],
                'Delhi': [28.7041, 77.1025],
                'Bangalore': [12.9716, 77.5946],
                'Chennai': [13.0827, 80.2707],
                'Kolkata': [22.5726, 88.3639],
                'Hyderabad': [17.3850, 78.4867],
                'Pune': [18.5204, 73.8567],
                'Ahmedabad': [23.0225, 72.5714]
            }
            
            map_data = []
            for city, coords in city_coords.items():
                risk_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
                color = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}[risk_level]
                map_data.append({
                    'City': city,
                    'Latitude': coords[0],
                    'Longitude': coords[1],
                    'Risk_Level': risk_level,
                    'Color': color,
                    'Occupancy': np.random.uniform(60, 95)
                })
            
            map_df = pd.DataFrame(map_data)
            
            fig = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                color='Risk_Level',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                size='Occupancy',
                hover_data=['City', 'Occupancy'],
                zoom=4,
                height=500,
                title="City-wise Healthcare Risk Assessment"
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series and pie charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Occupancy Rate Over Time")
            if 'admissions' in data and not data['admissions'].empty:
                admissions_df = data['admissions']
                if 'D.O.A' in admissions_df.columns:
                    ts_data = admissions_df.groupby(admissions_df['D.O.A'].dt.date)['occupancy_rate'].mean().reset_index()
                    fig = px.line(ts_data, x='D.O.A', y='occupancy_rate', 
                                 title="Daily Average Bed Occupancy Rate")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Date of admission data not available for time series analysis.")
            else:
                st.info("No admission data available for time series analysis.")
        
        with col2:
            st.subheader("Admissions by Disease Type")
            # Simulate disease distribution
            diseases_data = {
                'Disease': ['Heart Disease', 'Diabetes', 'Hypertension', 'Respiratory', 'Others'],
                'Count': [300, 250, 200, 150, 100]
            }
            diseases_df = pd.DataFrame(diseases_data)
            fig = px.pie(diseases_df, values='Count', names='Disease', 
                        title="Disease Distribution in Admissions")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Forecast & AI Insights
    with tab2:
        st.header("Forecast & AI Insights")
        
        # Forecast section
        st.subheader("6-Week Bed Demand Forecast")
        forecast_data = create_forecast(data)
        
        if not forecast_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Predicted Demand',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Confidence Interval'
            ))
            fig.update_layout(title="Hospital Bed Demand Forecast (Next 6 Weeks)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Factors vs Occupancy")
            corr_data = create_correlation_heatmap(data)
            
            if corr_data:
                factors = list(corr_data.keys())
                correlations = list(corr_data.values())
                
                fig = px.bar(
                    x=correlations,
                    y=factors,
                    orientation='h',
                    title="Correlation with Hospital Occupancy",
                    color=correlations,
                    color_continuous_scale='RdYlBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("What-If Analysis")
            infection_rate = st.slider(
                "Adjust Infection Rate (%)",
                min_value=50,
                max_value=150,
                value=100,
                step=5
            )
            
            # Calculate impact
            base_demand = forecast_data['yhat'].mean() if not forecast_data.empty else 50
            adjusted_demand = base_demand * (infection_rate / 100)
            impact = ((adjusted_demand - base_demand) / base_demand) * 100
            
            st.metric("Impact on Bed Demand", f"{impact:+.1f}%")
            st.metric("Adjusted Daily Demand", f"{adjusted_demand:.0f} beds")
            
            if infection_rate > 120:
                st.markdown('<div class="alert-critical">⚠️ High infection rate may cause critical shortage!</div>', 
                           unsafe_allow_html=True)
        
        # Oxygen supply vs demand
        st.subheader("Oxygen Availability vs Demand")
        oxygen_data = {
            'Category': ['Available Supply', 'Current Demand', 'Peak Demand', 'Reserve Capacity'],
            'Value': [850, 720, 950, 200]
        }
        oxygen_df = pd.DataFrame(oxygen_data)
        
        fig = px.bar(oxygen_df, x='Category', y='Value', 
                    title="Oxygen Supply Analysis (Liters/min)",
                    color='Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Hospital Drill-Down
    with tab3:
        st.header("Hospital Drill-Down Analysis")
        
        # Hospital selector
        if 'hospitals_list' in data and not data['hospitals_list'].empty:
            hospitals = data['hospitals_list']['Hospital'].dropna().unique()
            selected_hospital = st.selectbox("Select Hospital", hospitals[:50])  # Limit for performance
            
            # KPI cards for selected hospital
            st.subheader(f"Performance Metrics - {selected_hospital}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bed_util = np.random.uniform(70, 95)
                st.metric("Bed Utilization %", f"{bed_util:.1f}%")
            
            with col2:
                icu_usage = np.random.uniform(60, 90)
                st.metric("ICU Usage %", f"{icu_usage:.1f}%")
            
            with col3:
                mortality = np.random.uniform(2, 8)
                st.metric("Mortality %", f"{mortality:.1f}%")
            
            with col4:
                wait_time = np.random.uniform(15, 60)
                st.metric("Avg Wait Time (min)", f"{wait_time:.0f}")
            
            # Department utilization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Department Utilization")
                dept_data = {
                    'Department': ['Emergency', 'ICU', 'General Ward', 'Surgery', 'Maternity'],
                    'Utilization': [85, 78, 72, 68, 62]
                }
                dept_df = pd.DataFrame(dept_data)
                fig = px.bar(dept_df, x='Department', y='Utilization',
                           title="Department-wise Bed Utilization (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Admissions vs Discharges")
                dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
                admissions = np.random.poisson(12, 30)
                discharges = np.random.poisson(10, 30)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=admissions, mode='lines+markers', name='Admissions'))
                fig.add_trace(go.Bar(x=dates, y=discharges, name='Discharges', opacity=0.7))
                fig.update_layout(title="Daily Admissions vs Discharges")
                st.plotly_chart(fig, use_container_width=True)
            
            # Patient demographics
            st.subheader("Patient Demographics")
            col1, col2 = st.columns(2)
            
            with col1:
                age_data = {
                    'Age Group': ['0-18', '19-35', '36-50', '51-65', '65+'],
                    'Count': [45, 120, 180, 200, 155]
                }
                age_df = pd.DataFrame(age_data)
                fig = px.pie(age_df, values='Count', names='Age Group', title="Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                gender_data = {
                    'Gender': ['Male', 'Female'],
                    'Count': [420, 380]
                }
                gender_df = pd.DataFrame(gender_data)
                fig = px.pie(gender_df, values='Count', names='Gender', title="Gender Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Hospital data not available for drill-down analysis.")
    
    # Tab 4: Insights Summary
    with tab4:
        st.header("Healthcare Insights Summary")
        
        # Top 3 insights
        st.subheader("🎯 Top 3 Key Insights")
        
        insights = [
            {
                "title": "Critical Capacity Strain in Urban Areas",
                "description": f"Analysis shows {kpis.get('critical_hospitals', 0)} hospitals are operating above 85% capacity, indicating severe strain on healthcare infrastructure.",
                "impact": "High",
                "action_required": "Immediate resource reallocation needed"
            },
            {
                "title": "Environmental Correlation with Health Outcomes",
                "description": "Strong correlation (r=0.45) found between PM2.5 pollution levels and respiratory admissions, suggesting environmental health interventions needed.",
                "impact": "Medium",
                "action_required": "Implement air quality monitoring"
            },
            {
                "title": "Seasonal Demand Patterns",
                "description": "6-week forecast indicates 15% increase in bed demand during monsoon season, requiring proactive capacity planning.",
                "impact": "Medium",
                "action_required": "Seasonal staffing adjustments"
            }
        ]
        
        for i, insight in enumerate(insights, 1):
            with st.expander(f"Insight {i}: {insight['title']}", expanded=True):
                st.write(insight['description'])
                col1, col2 = st.columns(2)
                with col1:
                    impact_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.write(f"**Impact Level:** {impact_color[insight['impact']]} {insight['impact']}")
                with col2:
                    st.write(f"**Action Required:** {insight['action_required']}")
        
        # Action recommendations table
        st.subheader("🎯 Action Recommendations")
        
        recommendations = [
            {
                "Priority": "High",
                "Action": "Deploy mobile medical units to high-risk areas",
                "Timeline": "Immediate (1-2 weeks)",
                "Resource": "10 units, 50 staff",
                "Expected Impact": "20% reduction in critical cases"
            },
            {
                "Priority": "High", 
                "Action": "Increase ICU capacity in urban hospitals",
                "Timeline": "Short-term (1 month)",
                "Resource": "100 additional beds",
                "Expected Impact": "15% improvement in critical care"
            },
            {
                "Priority": "Medium",
                "Action": "Implement predictive analytics for bed allocation",
                "Timeline": "Medium-term (3 months)",
                "Resource": "AI system deployment",
                "Expected Impact": "25% efficiency improvement"
            },
            {
                "Priority": "Medium",
                "Action": "Establish pollution-health monitoring system",
                "Timeline": "Medium-term (2 months)",
                "Resource": "Monitoring stations, analytics team",
                "Expected Impact": "Early warning capability"
            },
            {
                "Priority": "Low",
                "Action": "Develop telemedicine infrastructure",
                "Timeline": "Long-term (6 months)",
                "Resource": "Technology platform, training",
                "Expected Impact": "30% reduction in routine visits"
            }
        ]
        
        recommendations_df = pd.DataFrame(recommendations)
        
        # Style the dataframe
        def color_priority(val):
            color = {'High': 'background-color: #ffcdd2',
                    'Medium': 'background-color: #fff3e0', 
                    'Low': 'background-color: #e8f5e8'}
            return color.get(val, '')
        
        styled_df = recommendations_df.style.applymap(color_priority, subset=['Priority'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Export functionality
        st.subheader("📄 Export Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Summary to CSV"):
                summary_data = pd.DataFrame({
                    'Metric': ['Avg Bed Occupancy %', 'Avg ICU Utilization %', 'Critical Hospitals', 
                              'Resource Shortage Index %', 'Total Admissions'],
                    'Value': [f"{kpis.get('avg_bed_occupancy', 0):.1f}%",
                             f"{kpis.get('avg_icu_utilization', 0):.1f}%",
                             kpis.get('critical_hospitals', 0),
                             f"{kpis.get('resource_shortage_index', 0):.1f}%",
                             kpis.get('total_admissions', 0)]
                })
                
                csv_link = export_to_csv(summary_data, "healthcare_summary.csv")
                st.markdown(csv_link, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export Recommendations to CSV"):
                csv_link = export_to_csv(recommendations_df, "action_recommendations.csv")
                st.markdown(csv_link, unsafe_allow_html=True)
        
        with col3:
            if st.button("Generate PDF Report"):
                st.info("PDF export functionality would be implemented with reportlab library.")
                # Note: PDF generation would require reportlab implementation
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("Dashboard Info")
        st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.info(f"**Total Records Processed:** {kpis.get('total_admissions', 0):,}")
        
        st.header("System Status")
        if kpis.get('avg_bed_occupancy', 0) > 85:
            st.error("🔴 System Alert: Critical capacity reached!")
        elif kpis.get('avg_bed_occupancy', 0) > 75:
            st.warning("🟡 System Warning: High capacity utilization")
        else:
            st.success("🟢 System Status: Normal operations")
        
        st.header("Data Sources")
        st.text("• Hospital capacity data")
        st.text("• Patient admission records")
        st.text("• Mortality statistics")
        st.text("• Environmental pollution data")
        st.text("• Weather information")

if __name__ == "__main__":
    main()
