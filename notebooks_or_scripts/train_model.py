import pandas as pd

train_data = pd.read_csv("../data/processed/train.csv")
test_data = pd.read_csv("../data/processed/test.csv")


X_train = train_data["cleaned_text"]
y_train = train_data["category"]

X_test = test_data["cleaned_text"]
y_test = test_data["category"]

print("Training requests:", len(X_train))
print("Testing requests:", len(X_test))

print("\nExample training text:")
print(X_train.head())

print("\nCorrect categories:")
print(y_train.head())
