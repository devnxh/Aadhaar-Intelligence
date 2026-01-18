# Quick Start Guide - Aadhaar Intelligence Platform

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (Optional)
If you don't have real datasets, generate sample data:
```bash
python utils/generate_sample_data.py
```

### Step 3: Run the Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Using the Dashboard

### Filters (Left Sidebar)
- **Date Range**: Select time period for analysis
- **State**: Filter by specific state or view all
- **District**: Filter by district (updates based on state selection)

### Tabs

1. **📊 Overview**
   - Key metrics cards (total enrolments, updates, UPI, districts)
   - Enrolment trends over time
   - Update type breakdown (demographic vs biometric)

2. **📈 Trends & Analysis**
   - State-wise Update Pressure Index heatmap
   - Top districts by various metrics
   - Customizable metric selection

3. **🔮 Forecasting**
   - Select target variable to forecast
   - Choose between ARIMA and Linear Regression
   - View 6-month forecasts with confidence intervals
   - Model performance metrics (MAE, RMSE, R²)

4. **⚠️ Anomaly Detection**
   - Statistical anomaly detection (Z-score method)
   - Anomalous patterns by state
   - Detailed anomaly records table

5. **💡 Insights**
   - Auto-generated key insights
   - Top performing states/districts
   - Recommendations for resource planning

---

## 📓 Running Notebooks

Execute notebooks in order:

```bash
jupyter notebook
```

1. `01_data_cleaning.ipynb` - Data preprocessing
2. `02_eda.ipynb` - Exploratory analysis (create this as needed)
3. `03_indicators.ipynb` - Calculate indicators (create this as needed)
4. `04_anomaly_detection.ipynb` - Detect anomalies (create this as needed)
5. `05_forecasting.ipynb` - Generate forecasts (create this as needed)

---

## 🔑 Key Indicators Explained

### Update Pressure Index (UPI)
```
UPI = Total Updates / Total Enrolment
```
- Higher values indicate more update demand per enrolment
- Use for resource allocation planning

### Age Transition Rate
```
ATR = Adult Biometric Updates / Child Enrolments
```
- Tracks child-to-adult transitions
- Indicates demographic shifts

### Growth Rates
- Period-over-period percentage change
- Identifies trends and patterns

---

## ⚠️ Troubleshooting

### "Error loading data"
- Ensure CSV files are in `data/` directory
- Check file names match: `enrolment.csv`, `demographic_updates.csv`, `biometric_updates.csv`

### "Module not found"
- Run: `pip install -r requirements.txt`
- Ensure virtual environment is activated

### Dashboard not loading
- Check if port 8501 is available
- Try: `streamlit run dashboard/app.py --server.port 8502`

### Forecasting errors
- Ensure sufficient historical data (at least 12 data points)
- Try switching from ARIMA to Linear method

---

## 💾 Data Requirements

### Enrolment Dataset
```csv
date,state,district,pincode,age_0_5,age_5_17,age_18_greater
2023-01-01,Maharashtra,Mumbai,400001,100,200,500
```

### Demographic Updates
```csv
date,state,district,pincode,demo_age_5_17,demo_age_17_
2023-01-01,Maharashtra,Mumbai,400001,50,100
```

### Biometric Updates
```csv
date,state,district,pincode,bio_age_5_17,bio_age_17_
2023-01-01,Maharashtra,Mumbai,400001,40,90
```

---

## 🎯 Next Steps

1. **Customize Analysis**: Modify notebooks for specific insights
2. **Add Indicators**: Extend `utils/indicators.py` with new metrics
3. **Enhance Dashboard**: Add new visualizations in `dashboard/app.py`
4. **Export Results**: Save insights and forecasts for reports

---

## 📞 Need Help?

- Check notebook comments and docstrings
- Review README.md for detailed documentation
- Examine utility module functions in `utils/`

---

**Built for Aadhaar Hackathon | Privacy-First | Decision Support**
