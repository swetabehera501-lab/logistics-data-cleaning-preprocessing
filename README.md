# Week 2 — Logistics Data Cleaning and Preprocessing

## Purpose
A reproducible Python pipeline showing data cleaning and preprocessing for shipment-level logistics data.

## Dataset
`raw_logistics_data.csv` contains 30 **synthetic demonstration records**, not real company data or a downloaded public dataset. It includes order ID, date, origin/destination, mode, planned/actual transit hours, planned/actual costs, and vehicle utilization.

## Issues simulated
Missing numeric/category values, duplicate order ID, inconsistent text case/whitespace, negative planned time, utilization above 100%, and an extreme transit-time value.

## Run
```powershell
python -m pip install -r requirements.txt
python preprocess.py
```

## Outputs
- `outputs/cleaned_logistics_data.csv`
- `outputs/outlier_review.csv`
- `outputs/preprocessing_audit.csv`

## Methods
Text standardization, type conversion, duplicate removal, domain validation, IQR outlier flagging, median/mode imputation, derived features, and Min-Max scaling.

## Caution
Outliers are flagged, not automatically deleted. Imputation/scaling choices are illustrative. In predictive modeling, fit imputers/scalers on training data only to avoid data leakage. Confirm suspicious records with a data owner before altering production data.
