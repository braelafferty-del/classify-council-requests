import pandas as pd
from sklearn.model_selection import train_test_split

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


# Step 5: Choose the input and target.
X = df["cleaned_text"]  # Text used to make predictions.
y = df["category"]      # Correct category to learn.

# Step 6: Use 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining requests:", len(X_train))
print("Testing requests:", len(X_test))

print("\nTest categories:")
print(y_test.value_counts())
