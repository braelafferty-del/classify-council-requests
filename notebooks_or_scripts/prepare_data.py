import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset from the data folder.
df = pd.read_csv(
    "../data/raw/BCIT601_starter_service_requests_dataset.csv"
)

# Step 2: Show the first five rows.
print(df.head())

# Step 3: Show the number of rows and columns.
print("Shape:", df.shape)

print("Number of requests by category")
print(df["category"].value_counts())

print("Missing values before cleaning")
print(df.isnull().sum())

# Clean text by replacing missing text, converting to lowercase and removing extra spaces.
df["cleaned_text"] = df["request_text"].fillna("unknown request")
df["cleaned_text"] = df["cleaned_text"].str.lower().str.strip()
df["cleaned_text"] = df["cleaned_text"].str.split().str.join(" ")

print("Duplicate request texts before cleaning:")
print(df.duplicated(subset=["cleaned_text"]).sum())

# Remove duplicate records.
df = df.drop_duplicates(subset=["cleaned_text"])

print("Missing values after cleaning")
print(df.isnull().sum())

print("Duplicate request texts after cleaning:")
print(df.duplicated(subset=["cleaned_text"]).sum())

print("Rows after cleaning:", len(df))
print(df["category"].value_counts())

# X holds the features used to make the prediction.
X = df["cleaned_text"]
y = df["category"]

# test_size=0.2 reserves 20% of the rows for testing (an 80/20
# split). random_state=42 makes the split reproducible, so everyone
# who runs this code gets the same training/testing rows.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("Test categories:")
print(y_test.value_counts())

Path("../data/processed").mkdir(parents=True, exist_ok=True)

# Save a cleaned copy.
df.to_csv("../data/processed/cleaned_requests.csv", index=False)

train_data = df.loc[X_train.index]
test_data = df.loc[X_test.index]

train_data.to_csv("../data/processed/train.csv", index=False)
test_data.to_csv("../data/processed/test.csv", index=False)

print("Prepared data saved.")
