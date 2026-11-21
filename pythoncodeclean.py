import pandas as pd
import numpy as np

input_file = "shipments and transportation project.csv"
output_file = "cleaned_shipments.csv"

df = pd.read_csv("shipments and transportation project.csv")
print(f"Loaded {len(df)} rows. Starting data cleaning...")


# ==============================================================================
# 1. REMOVE DUPLICATES & CLEAN TEXT WHITESPACE
# ==============================================================================
# Standardize shipment_id to uppercase and remove exact duplicate rows
df['shipment_id'] = df['shipment_id'].astype(str).str.strip().str.upper()
df = df.drop_duplicates(subset=['shipment_id'], keep='first').copy()

# Strip extra spaces from all text columns and replace fake nulls with NaN
text_cols = df.select_dtypes(include=['object', 'string']).columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 
                               'NA': np.nan, 'N/A': np.nan, 'null': np.nan, 
                               '?': np.nan, '0': np.nan, 'TBD': np.nan, 
                               'Unknown': np.nan, 'unknown': np.nan, 'driver': np.nan})


# ==============================================================================
# 2. STANDARDIZE CITIES, CARRIERS, MODES, VEHICLES & STATUSES
# ==============================================================================
# Fix city typos and legacy names
city_mapping = {
    'Bombay': 'Mumbai',
    'Bangalore': 'Bengaluru',
    'Madras': 'Chennai',
    'Calcutta': 'Kolkata',
    'Dehli': 'Delhi',
    'New Delhi': 'Delhi',
    'Hydrabad': 'Hyderabad',
    'Ahmedabd': 'Ahmedabad'
}
for col in ['origin_city', 'destination_city']:
    df[col] = df[col].str.title().replace(city_mapping)

# Fix carrier naming inconsistencies
carrier_mapping = {
    'D.T.D.C': 'DTDC',
    'Dtdc': 'DTDC',
    'Blue Dart': 'BlueDart',
    'Bluedart': 'BlueDart',
    'Ecomexpress': 'Ecom Express',
    'Delhivary': 'Delhivery'
}
df['carrier'] = df['carrier'].str.title().replace(carrier_mapping)

# Standardize shipping mode & status
df['shipping_mode'] = df['shipping_mode'].str.title()
df['status'] = df['status'].str.title()

# Standardize vehicle types
vehicle_mapping = {
    'Minivan': 'Mini Van',
    '16T Truck': 'Truck 16T',
    '10T Truck': 'Truck 10T',
    'Truck-10T': 'Truck 10T',
    'Cont. Ship': 'Container Ship'
}
df['vehicle_type'] = df['vehicle_type'].str.title().replace(vehicle_mapping)

# Standardize payment methods
payment_mapping = {
    'Cod': 'COD',
    'C.O.D': 'COD',
    'Cash On Delivery': 'COD',
    'Upi': 'UPI',
    'U.P.I': 'UPI',
    'Pre-Paid': 'Prepaid',
    'Net 30': 'Credit (Net 30)',
    'Credit Net30': 'Credit (Net 30)',
    'Credit': 'Credit (Net 30)',
    'Neft': 'NEFT',
    'Rtgs': 'RTGS'
}
df['payment_method'] = df['payment_method'].str.title().replace(payment_mapping)


# ==============================================================================
# 3. CLEAN NUMERIC COLUMNS (WEIGHT, COST, DELIVERY DAYS)
# ==============================================================================
# Clean weight_kg: remove 'kg', convert to float, and turn negative values into NaN
df['weight_kg'] = (
    df['weight_kg']
    .astype(str)
    .str.replace('kg', '', case=False, regex=False)
    .str.strip()
)
df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
df.loc[df['weight_kg'] <= 0, 'weight_kg'] = np.nan

# Clean freight_cost: remove '₹', 'Rs.', commas, and turn negative costs into NaN
df['freight_cost'] = (
    df['freight_cost']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace('Rs.', '', case=False, regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df['freight_cost'] = pd.to_numeric(df['freight_cost'], errors='coerce')
df.loc[df['freight_cost'] <= 0, 'freight_cost'] = np.nan

# Clean delivery_days: remove 'days', convert to numeric, remove negative/99 outliers
df['delivery_days'] = (
    df['delivery_days']
    .astype(str)
    .str.replace('days', '', case=False, regex=False)
    .str.strip()
)
df['delivery_days'] = pd.to_numeric(df['delivery_days'], errors='coerce')
df.loc[(df['delivery_days'] < 0) | (df['delivery_days'] >= 90), 'delivery_days'] = np.nan


# ==============================================================================
# 4. FIX DATES & RECALCULATE DELIVERY DAYS
# ==============================================================================
# Replace invalid dates like '0000-00-00' or '31/02/2025' before parsing
for col in ['ship_date', 'delivery_date']:
    df[col] = df[col].replace({'0000-00-00': np.nan, '31/02/2025': np.nan})
    df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')

# If delivery_date is earlier than ship_date (data entry swap/error), set delivery_date to NaT
invalid_dates = df['delivery_date'] < df['ship_date']
df.loc[invalid_dates, 'delivery_date'] = pd.NaT

# Fill missing delivery_days using the actual difference between delivery_date and ship_date
calculated_days = (df['delivery_date'] - df['ship_date']).dt.days
df['delivery_days'] = df['delivery_days'].fillna(calculated_days)


# ==============================================================================
# 5. STANDARDIZE DRIVER DETAILS
# ==============================================================================
df['driver_id'] = df['driver_id'].str.upper()
df['driver_name'] = df['driver_name'].str.title()


# ==============================================================================
# 6. EXPORT CLEANED DATA
# ==============================================================================
df.to_csv(output_file, index=False)
print(f"Cleaning complete! Saved {len(df)} clean rows to '{output_file}'.")
