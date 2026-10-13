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

# Step 3: Clean the request text.
# Keep the original text and create a cleaned version.
df["cleaned_text"] = df["request_text"].fillna("")
df["cleaned_text"] = df["cleaned_text"].str.lower()

# Replace punctuation with spaces, then remove extra whitespace.
df["cleaned_text"] = df["cleaned_text"].str.replace(
    r"[^\w\s]", " ", regex=True
)
df["cleaned_text"] = df["cleaned_text"].str.split().str.join(" ")

# Remove requests that contain no usable text.
df = df[df["cleaned_text"] != ""].copy()

# Step 4: Remove repeated request texts.
print("\nRepeated request texts:")
print(df.duplicated(subset=["cleaned_text"]).sum())

df = df.drop_duplicates(subset=["cleaned_text"]).copy()

print("\nRows after cleaning:")
print(len(df))

print("\nCategories after cleaning:")
print(df["category"].value_counts())
