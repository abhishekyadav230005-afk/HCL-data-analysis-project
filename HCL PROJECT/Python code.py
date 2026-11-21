import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. LOAD DATA
file_name = "shipments and transportation project.csv"
df = pd.read_csv(file_name)
print("Raw data shape:", df.shape)

# 2. CLEAN DATA
df = df.drop_duplicates(subset=["shipment_id"]).copy()

# Clean text columns and handle placeholder values like '?'
text_cols = [
    "origin_city",
    "destination_city",
    "carrier",
    "shipping_mode",
    "vehicle_type",
    "payment_method",
    "status",
]
for col in text_cols:
  df[col] = df[col].astype(str).str.strip()
  df.loc[df[col].isin(["nan", "None", "?", ""]), col] = np.nan

# Standardize city, carrier, mode, status, and payment labels
city_map = {
    "madras": "Chennai",
    "ahmedabd": "Ahmedabad",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "bengaluru": "Bengaluru",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
}
for col in ["origin_city", "destination_city"]:
  df[col] = df[col].apply(
      lambda x: city_map.get(x.lower(), x.title()) if pd.notna(x) else x
  )

carrier_map = {
    "dtdc": "DTDC",
    "d.t.d.c": "DTDC",
    "gati": "Gati",
    "bluedart": "BlueDart",
    "blue dart": "BlueDart",
    "delhivery": "Delhivery",
    "delhivary": "Delhivery",
    "ecom express": "Ecom Express",
    "ecomexpress": "Ecom Express",
}
df["carrier"] = df["carrier"].apply(
    lambda x: carrier_map.get(x.lower(), x.title()) if pd.notna(x) else x
)
df["shipping_mode"] = df["shipping_mode"].str.title()
df["status"] = df["status"].str.title()

payment_map = {
    "upi": "UPI",
    "u.p.i": "UPI",
    "cod": "COD",
    "c.o.d": "COD",
    "cash on delivery": "COD",
    "prepaid": "Prepaid",
    "pre-paid": "Prepaid",
    "credit (net 30)": "Credit (Net 30)",
    "net 30": "Credit (Net 30)",
    "credit": "Credit (Net 30)",
    "bank transfer": "Bank Transfer",
}
df["payment_method"] = df["payment_method"].apply(
    lambda x: payment_map.get(x.lower(), x.title()) if pd.notna(x) else x
)

# Remove units/symbols (like 'kg', 'Rs.', '₹', ',', 'days') and convert to numbers
for col in ["weight_kg", "freight_cost", "delivery_days"]:
  df[col] = pd.to_numeric(
      df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce"
  )

# Remove negative/invalid values and cap extreme outliers at the 99th percentile
df.loc[df["weight_kg"] <= 0, "weight_kg"] = np.nan
df.loc[df["freight_cost"] <= 0, "freight_cost"] = np.nan
df.loc[(df["delivery_days"] <= 0) | (df["delivery_days"] > 60), "delivery_days"] = (
    np.nan
)

df["weight_kg"] = df["weight_kg"].clip(upper=df["weight_kg"].quantile(0.99))
df["freight_cost"] = df["freight_cost"].clip(
    upper=df["freight_cost"].quantile(0.99)
)

# Parse dates and estimate missing delivery_days from ship_date & delivery_date
df["ship_date"] = pd.to_datetime(
    df["ship_date"], format="mixed", dayfirst=True, errors="coerce"
)
df["delivery_date"] = pd.to_datetime(
    df["delivery_date"], format="mixed", dayfirst=True, errors="coerce"
)

calc_days = (df["delivery_date"] - df["ship_date"]).dt.days
df["delivery_days"] = df["delivery_days"].fillna(
    calc_days.where((calc_days > 0) & (calc_days <= 60))
)

# Fill remaining missing values with median (numbers) or mode (categories)
for col in ["weight_kg", "freight_cost", "delivery_days"]:
  df[col] = df[col].fillna(df[col].median())

for col in ["destination_city", "vehicle_type", "payment_method"]:
  df[col] = df[col].fillna(df[col].mode()[0])

# Create helpful logistics features
df["cost_per_kg"] = (df["freight_cost"] / df["weight_kg"]).round(2)
df["route"] = df["origin_city"] + " -> " + df["destination_city"]

# EXPORT CLEANED DATASET
df.to_csv("cleaned_shipments.csv", index=False)
print("Cleaned data shape:", df.shape)
print("Saved cleaned dataset to 'cleaned_shipments.csv'")

# 3. TRANSPORTATION & LOGISTICS ANALYSIS
print("\n--- 1. SHIPMENT STATUS & DELAY RATE ---")
print(df["status"].value_counts())
print(f"Delivery Rate: {(df['status'] == 'Delivered').mean() * 100:.1f}%")
print(f"Delay Rate:    {(df['status'] == 'Delayed').mean() * 100:.1f}%")

print("\n--- 2. CARRIER PERFORMANCE ---")
carrier_perf = (
    df.groupby("carrier")
    .agg(
        shipments=("shipment_id", "count"),
        avg_cost=("freight_cost", "mean"),
        avg_days=("delivery_days", "mean"),
        delay_pct=("status", lambda s: (s == "Delayed").mean() * 100),
    )
    .round(2)
)
print(carrier_perf)

print("\n--- 3. SHIPPING MODE EFFICIENCY ---")
mode_perf = (
    df.groupby("shipping_mode")
    .agg(
        shipments=("shipment_id", "count"),
        avg_weight_kg=("weight_kg", "mean"),
        avg_cost_per_kg=("cost_per_kg", "mean"),
        avg_days=("delivery_days", "mean"),
    )
    .round(2)
)
print(mode_perf)

print("\n--- 4. TOP 5 BUSIEST ROUTES ---")
print(df["route"].value_counts().head(5))

print("\n--- 5. PAYMENT METHOD BREAKDOWN ---")
print(df["payment_method"].value_counts())

# 4. VISUALIZATION (2x2 CHART DASHBOARD)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Shipment & Transportation Analysis Overview", fontsize=14, fontweight="bold"
)

df["status"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Shipments by Status")
axes[0, 0].set_ylabel("Number of Shipments")
axes[0, 0].tick_params(axis="x", rotation=20)

carrier_perf["avg_cost"].sort_values().plot(
    kind="barh", ax=axes[0, 1], color="teal"
)
axes[0, 1].set_title("Average Freight Cost by Carrier")
axes[0, 1].set_xlabel("Average Freight Cost")

mode_perf["avg_days"].sort_values().plot(kind="bar", ax=axes[1, 0], color="coral")
axes[1, 0].set_title("Average Delivery Days by Mode")
axes[1, 0].set_ylabel("Average Days")
axes[1, 0].tick_params(axis="x", rotation=0)

df["route"].value_counts().head(5).sort_values().plot(
    kind="barh", ax=axes[1, 1], color="slateblue"
)
axes[1, 1].set_title("Top 5 Shipment Routes")
axes[1, 1].set_xlabel("Number of Shipments")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("shipment_analysis_dashboard.png", dpi=150)
plt.show()