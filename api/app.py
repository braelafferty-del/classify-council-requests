# Flask API 

from flask import Flask, request 
import joblib
app = Flask(__name__)

# Load model and vectoriser 
model = joblib.load("../model/model.joblib")
vectoriser = joblib.load("../model/vectoriser.joblib")

# /Health route to show the app is working
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

# / Categories endpoint to show all the endpoints 
@app.route("/categories", methods=["GET"])
def categories():
    return {"categories": model.classes_.tolist()}

# /Predict Route - Uses the model thats already compiled and predicts based on the text given
# Used by in app ?text=
@app.route("/predict", methods=["GET"])
def predict_from_browser():
    text = request.args.get("text", "")
    text = " ".join(text.lower().split())

    if text == "":
        return {"error": "use predict/?text=text-goes-here"}, 400

    features = vectoriser.transform([text])

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)

    return {
        "request_text": text,
        "predicted_category": str(predictions[0]),
        "confidence": float(confidence[0])
    }
if __name__ == "__main__":
    app.run(debug=True)
