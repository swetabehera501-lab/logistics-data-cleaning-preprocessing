"""Week 2 logistics preprocessing demonstration; input data is synthetic."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "raw_logistics_data.csv")
initial_rows = len(df)

# Standardize category strings and parse types
for col in ["origin", "destination", "transport_mode"]:
    df[col] = df[col].astype("string").str.strip().str.title()
df["order_id"] = df["order_id"].astype("string").str.strip()
df["dispatch_date"] = pd.to_datetime(df["dispatch_date"], errors="coerce")
num = ["planned_transit_hours","actual_transit_hours","planned_cost_inr","actual_cost_inr","vehicle_utilization_pct"]
for col in num:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove repeated order IDs; log the number found
duplicates = int(df["order_id"].duplicated().sum())
df = df.drop_duplicates(subset=["order_id"], keep="first").copy()

# Treat impossible values as missing so they can be reviewed/imputed
for col in ["planned_transit_hours","actual_transit_hours","planned_cost_inr","actual_cost_inr"]:
    df.loc[df[col] < 0, col] = float("nan")
df.loc[~df["vehicle_utilization_pct"].between(0,100), "vehicle_utilization_pct"] = float("nan")

# Flag outliers with the 1.5 x IQR rule; don't delete automatically
q1, q3 = df["actual_transit_hours"].quantile([0.25,0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
df["actual_transit_outlier_flag"] = df["actual_transit_hours"].notna() & ~df["actual_transit_hours"].between(lo,hi)

# Demonstration imputation: median for numeric, mode for categorical
for col in num:
    med = df[col].median()
    if pd.notna(med): df[col] = df[col].fillna(med)
for col in ["origin","destination","transport_mode"]:
    modes = df[col].mode(dropna=True)
    if not modes.empty: df[col] = df[col].fillna(modes.iloc[0])

# Derived features and Min-Max normalized copies; preserve original units
df["delay_hours"] = df["actual_transit_hours"] - df["planned_transit_hours"]
df["cost_overrun_inr"] = df["actual_cost_inr"] - df["planned_cost_inr"]
scale_cols = ["planned_transit_hours","actual_transit_hours","planned_cost_inr","actual_cost_inr","vehicle_utilization_pct"]
scaled = MinMaxScaler().fit_transform(df[scale_cols])
for i,col in enumerate(scale_cols): df[col+"_minmax"] = scaled[:,i]

df.to_csv(OUT/"cleaned_logistics_data.csv", index=False)
df.loc[df["actual_transit_outlier_flag"]].to_csv(OUT/"outlier_review.csv", index=False)
audit = pd.DataFrame([
    ["Rows before cleaning", initial_rows],
    ["Duplicate order IDs found", duplicates],
    ["Rows after duplicate removal", len(df)],
    ["Missing cells after preprocessing", int(df.isna().sum().sum())],
    ["Transit-time outliers flagged", int(df["actual_transit_outlier_flag"].sum())],
], columns=["check","value"])
audit.to_csv(OUT/"preprocessing_audit.csv", index=False)
print("Preprocessing completed.")
print(audit.to_string(index=False))
print("Outputs saved to:", OUT)
print("Note: dataset is synthetic demonstration data, not real company data.")
