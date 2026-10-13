import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


train_data = pd.read_csv("../data/processed/train.csv")
test_data = pd.read_csv("../data/processed/test.csv")


X_train = train_data["cleaned_text"]
y_train = train_data["category"]

X_test = test_data["cleaned_text"]
y_test = test_data["category"]

# print("Training requests:", len(X_train))
# print("Testing requests:", len(X_test))

# print("Example training text:")
# print(X_train.head())

# print("Correct categories:")
# print(y_train.head())

vectoriser = TfidfVectorizer()
X_train_tfidf = vectoriser.fit_transform(X_train)
X_test_tfidf = vectoriser.transform(X_test)

# print("\nTraining feature table:", X_train_tfidf.shape)
# print("Testing feature table:", X_test_tfidf.shape)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

probabilities = model.predict_proba(X_test_tfidf)

confidence = probabilities.max(axis=1)

results = pd.DataFrame({
    "request_text": test_data["request_text"],
    "true_label": y_test,
    "predicted_label": predictions,
    "confidence": confidence
})

print("\nFive example predictions:")
print(results.head(5).to_string(index=False))

Path("../model").mkdir(parents=True, exist_ok=True)
Path("../outputs").mkdir(parents=True, exist_ok=True)

# Save model 
joblib.dump(model, "../model/model.joblib")
joblib.dump(vectoriser, "../model/vectoriser.joblib")

results.to_csv("../outputs/test_predictions.csv", index=False)

results.head(5).to_csv(
    "../outputs/five_example_predictions.csv",
    index=False
)
