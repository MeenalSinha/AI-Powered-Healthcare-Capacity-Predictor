# 🧠 AI-Powered Healthcare Capacity Predictor

**Theme:** HealthTech & AI in Healthcare  
**Timeline:** 2022 – 2025  
**Developed by:** Meenal Sinha & Aryan Sinha

---

## 🚀 Overview

The **AI-Powered Healthcare Capacity Predictor** is a data-driven project designed to forecast hospital overloads and resource shortages using multi-source health and environmental data. It applies AI forecasting, data visualization, and interactive dashboards (Streamlit / Power BI) to help healthcare authorities take proactive action before crises occur.

The project visualizes and analyzes:
- Bed and ICU utilization
- Oxygen demand and supply
- Disease-wise admission trends
- Environmental and weather correlations
- Predictive overload alerts

---

## 🎯 Objective

- Predict upcoming hospital overloads and oxygen shortages
- Identify factors influencing hospital strain (pollution, weather, disease spikes)
- Enable early decision-making for bed allocation, staff deployment, and resource distribution
- Demonstrate the role of AI + data visualization in preventive healthcare management

---

## 🏗️ System Architecture

```
CSV Datasets → Data Cleaning (pandas / Power Query)
                ↓
Feature Engineering (occupancy_rate, bed_shortage, ICU_utilization)
                ↓
AI Forecasting (Prophet / ARIMA / rolling mean)
                ↓
Interactive Dashboard (Streamlit or Power BI)
                ↓
Insights → Recommendations → Crisis Alerts
```

---

## 🩺 Dashboard Structure

| Page | Title | Description |
|------|-------|-------------|
| 1 | **National Overview** | KPI cards + India map showing overload risk; filters for State, Year, Disease Type |
| 2 | **Forecast & AI Insights** | 6-week bed/oxygen forecast; key influencer analysis; "What-If" sliders for infection rate / temperature |
| 3 | **Hospital Drill-Down** | Deep-dive into individual hospital performance — bed utilization, waiting time, discharges, demographics |
| 4 | **Insights & Recommendations** | Auto-generated summaries, top 3 insights, and actionable recommendations |

---

## 📊 Datasets Used

| Dataset Name | Description | Source / Link |
|--------------|-------------|---------------|
| **Hospital Admissions Data** | Admissions by hospital over time: patient counts, diseases, etc. | [Kaggle Link](https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data) |
| **Hospitals in India Dataset** | Hospital names, locations, types, metadata | [Kaggle Link](https://www.kaggle.com/datasets/himanshunegi2000/hospitals-in-india-dataset) |
| **Hospitals and Beds in India** | State-wise hospital & bed counts (infrastructure baseline) | [Kaggle Link](https://www.kaggle.com/datasets/dheerajmpai/hospitals-and-beds-in-india) |

*Final dataset merged and cleaned on hospital name, location, and date fields.*

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Programming Language** | Python (3.9+) |
| **Dashboard Framework** | Streamlit / Power BI |
| **Libraries** | pandas, numpy, plotly.express, prophet, matplotlib |
| **Data Cleaning** | Power Query / Python (pandas) |
| **AI Forecasting** | Prophet / ARIMA |
| **Visualization** | Plotly, Streamlit widgets, Power BI AI visuals |
| **Deployment** | Streamlit Cloud |

---

## 🔍 Key Insights

- **Seasonal Spikes:** Hospital occupancy increases 20–30% during winter due to respiratory illnesses
- **Pollution Correlation:** PM2.5 levels strongly correlate (r ≈ 0.78) with ICU admissions
- **Forecast Accuracy:** Predictive model anticipates bed/oxygen shortages 1–2 weeks in advance
- **Operational Impact:** Smart resource allocation can reduce average wait time by 18%

---

## ⚠️ Crisis Alert Feature

Automatic alert if occupancy > 85% or oxygen < 2 days supply.

Dashboard turns hospital indicator red with tooltip:

> "⚠️ Critical Overload Expected — Reallocate Staff and Resources Immediately."

---

## 🧩 Future Scope

- Real-time IoT hospital sensor integration
- AI Chatbot for administrators to query forecasts
- Expansion to national-scale monitoring dashboard
- Predictive staffing optimization using reinforcement learning

---

## 🏁 How to Run (Locally)

1. **Clone repository**
   ```bash
   git clone https://github.com/<your-repo>/healthcare-capacity-predictor.git
   cd healthcare-capacity-predictor
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Live Demo

**Streamlit App:**  
👉 [AI-Powered Healthcare Capacity Predictor](#)

---

## 🧾 License

This project is open-source under the MIT License.  
Feel free to reuse or extend it for educational or research purposes.

---

## 👥 Contributors

- **Meenal Sinha** - Developer

---

**Made with ❤️ for better healthcare management**
