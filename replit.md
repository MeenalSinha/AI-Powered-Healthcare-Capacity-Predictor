# AI-Powered Healthcare Capacity Predictor

## Overview
An interactive Streamlit dashboard for healthcare capacity prediction and analysis, combining patient admission data, hospital infrastructure statistics, and environmental factors to forecast bed demand and identify resource constraints.

## Recent Changes (October 15, 2025)
- **Data Integration**: Integrated 6 CSV datasets (hospitals, admissions, mortality, pollution)
- **Accurate Metrics Calculation**: Fixed bed occupancy to use peak-based capacity estimation (eliminates division by national totals)
- **Transparent Scope Communication**: Dashboard clearly labeled as analyzing reference hospital with national context separated
- **Complete Dashboard**: Built 4-tab interface with forecasting, drill-down, and insights
- **Export Functionality**: Added CSV and PDF report generation with ReportLab
- **Prophet Forecasting**: Integrated time-series forecasting for 6-week bed demand prediction
- **Data Integrity Fixes**: Replaced misleading national risk map with hospital distribution visualization

## Project Architecture

### Data Sources
1. **Hospitals_and_Beds_statewise** - State-level hospital and bed counts
2. **Hospitals_Rural_Urban** - Rural/urban bed distribution by state
3. **HospitalsInIndia** - List of 1,350+ hospitals with locations
4. **HDHI Admission data** - 15,759 patient admission records (2017-2019)
5. **HDHI Mortality Data** - 361 mortality records
6. **HDHI Pollution Data** - Daily environmental metrics (739 records)

### Data Integration Approach

**Important**: The admission/mortality data comes from a single reference hospital (HDHI) and contains:
- Patient demographics (age, gender, rural/urban)
- Medical conditions (heart disease, diabetes, hypertension)
- ICU utilization and duration of stay
- Temporal data (admission/discharge dates)

The dashboard combines this patient-level data with national hospital infrastructure statistics to provide:
- **Actual patient outcomes** from the reference hospital
- **National capacity context** from state-level bed counts
- **Environmental correlations** via pollution data merge on dates

This hybrid approach provides real patient insights while demonstrating national-scale analytics capabilities.

### Key Features

#### Tab 1: National Overview
- Year/disease filters (state filter for exploration only)
- Reference hospital KPIs: bed occupancy %, ICU utilization, critical load days %
- Hospital distribution map showing national infrastructure
- Time-series occupancy trends from reference hospital
- Disease distribution analysis from patient records

#### Tab 2: Forecast & AI Insights
- 6-week bed demand forecasting (Prophet or statistical fallback)
- Environmental factor correlation analysis (PM2.5, temperature vs admissions)
- What-if analysis for infection rate scenarios
- ICU capacity projections

#### Tab 3: Hospital Drill-Down
- Hospital selection by state
- Facility-specific KPIs (bed utilization, ICU usage, mortality %)
- Department utilization charts
- Patient demographics (age/gender distribution)
- Admissions vs discharges trends

#### Tab 4: Insights Summary
- Auto-generated top 3 insights from data analysis
- Priority-based action recommendations
- CSV export for insights and metrics
- PDF report generation for executive review

### Technical Stack
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Forecasting**: Prophet (Facebook's time-series forecasting)
- **PDF Generation**: ReportLab
- **Caching**: @st.cache_data for performance optimization

### Computed Metrics
- **Bed Occupancy Rate**: Daily occupied beds / estimated hospital capacity × 100
  - Hospital capacity estimated from peak occupancy / 0.85 (assumes peak = 85% capacity)
- **ICU Utilization**: ICU days / total hospital days × 100
- **Resource Shortage Index**: % of days exceeding 85% capacity
- **Mortality Rate**: Deaths / total admissions × 100
- **Critical Load Days**: % of days above 85% occupancy threshold

### Performance Optimizations
- Cached data loading with @st.cache_data
- Column-selective CSV reading
- Efficient datetime operations with pandas
- Conditional Prophet loading (fallback to statistical methods)

### Known Limitations
1. **Single Hospital Patient Data**: Admission records are from one hospital, not hospital-specific for the full list
2. **State Assignment**: Admissions don't have state identifiers - metrics are aggregated nationally then distributed for visualization
3. **Simplified Correlations**: Environmental correlations use date-based joins without geographic specificity
4. **Forecast Limitations**: Prophet requires 10+ data points; falls back to rolling mean for sparse data

### Future Enhancements (Next Phase)
- Real-time data integration with hospital APIs
- Advanced ML models (XGBoost, LSTM) for multi-week forecasting
- User accounts with saved preferences
- Automated email/SMS alerts for critical thresholds
- Comparative benchmarking tools across regions

## Setup Instructions

### Requirements
- Python 3.11+
- Dependencies: streamlit, pandas, numpy, plotly, prophet, reportlab, scikit-learn

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

### Configuration
Server settings are in `.streamlit/config.toml`:
- Port: 5000
- Address: 0.0.0.0
- Headless mode: enabled

## Data Flow

1. **Load Phase**: All CSV files loaded with date standardization
2. **Processing Phase**: 
   - Calculate daily bed occupancy from admission durations
   - Merge pollution data on dates
   - Compute ICU utilization from patient records
3. **Aggregation Phase**:
   - State-level bed statistics from infrastructure data
   - Patient outcomes from admission data
   - Environmental correlations from merged datasets
4. **Presentation Phase**:
   - Interactive filters and visualizations
   - Real-time metric updates
   - Export generation (CSV/PDF)

## Code Structure

- `load_and_process_data()` - Loads and cleans all CSV files
- `compute_metrics_and_merge()` - Calculates occupancy, ICU usage, correlations
- `create_forecast()` - Generates 6-week demand predictions
- `calculate_correlations()` - Environmental factor analysis
- `export_to_csv()` / `export_to_pdf()` - Report generation
- `main()` - Streamlit app with 4-tab interface

## User Experience

The dashboard is designed for:
- **Healthcare Administrators**: Monitor capacity and plan resources
- **Policy Makers**: Identify critical regions and allocate funding
- **Researchers**: Analyze environmental health impacts
- **Operations Teams**: Track daily metrics and respond to alerts
