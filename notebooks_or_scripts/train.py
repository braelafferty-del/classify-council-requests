
import pandas as pd

# Step 1: Load the council service requests.
df = pd.read_csv(
    "../data/raw/BCIT601_starter_service_requests_dataset.csv"
)

# Step 2: Inspect the dataset.
print("First five rows:")
print(df.head())

print("\nNumber of rows and columns:")
print(df.shape)

print("\nRequests in each category:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# OUTPUT First five rows:
#    request_id                                       request_text category        suburb priority date_reported
# 0           1  Large pothole outside 21 Colombo Street causin...     road      Sydenham     high    2026-05-01
# 1           2  Glass and loose gravel left on cycle lane near...     road     Riccarton   medium    2026-05-01
# 2           3  Traffic light at Moorhouse Avenue not changing...     road  Central City     high    2026-05-02
# 3           4  Deep crack forming on footpath beside the prim...     road        Hornby   medium    2026-05-02
# 4           5    Street sign has fallen over after strong winds.     road  New Brighton   medium    2026-05-03

# Number of rows and columns:
# (80, 6)

# Requests in each category:
# category
# road      20
# waste     20
# water     20
# safety    20
# Name: count, dtype: int64

# Missing values:
# request_id       0
# request_text     0
# category         0
# suburb           0
# priority         0
# date_reported    0
# dtype: int64
