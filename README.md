# Afficionado Coffee Roasters: Demand Forecasting

Hourly and daily sales forecasting with peak-demand detection for three coffee stores (Jan-Jun 2023). Unified Mentor project by Aditya Gupta.

## Contents
- forecasting.ipynb: full analysis (cleaning, EDA, features, models, peak detection, July forecast)
- app.py: Streamlit dashboard
- figures/: charts used in the paper
- *.csv: generated outputs used by the dashboard
- Afficionado_Forecasting_Research_Paper.docx: research paper and executive summary

## Run the dashboard
pip install pandas numpy plotly streamlit
streamlit run app.py

## Key results
- Best daily models: SARIMA and Prophet (MAPE about 14%)
- Hourly Gradient Boosting: WAPE 30.2% vs 45.7% baseline
- Peak capture 91.1% with 37.9% false alarms
- July 2023 forecast: about 174,000 total revenue across the 3 stores