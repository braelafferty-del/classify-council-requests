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


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    # If no data is sent
    if data is None:
        return {"error": "Please send valid JSON."}, 400
    if not isinstance(data, dict):
        return {
            "error": 'Send a JSON object like {"request_text": "missed rubbish collection"}.'
        }, 400
    text = data.get("request_text")
    # If request_text is empty error catch 
    if not isinstance(text, str) or text.strip() == "":
        return {"error": "request_text must contain some text."}, 400
    

    text = " ".join(text.lower().split())
    # Put the cleaned text to numbers TF-IDF
    features = vectoriser.transform([text])
    
    # Uses the model and use predict / predict_proba based on the numbers 
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = probabilities.max(axis=1)
    # Returns back to user 
    return {
        "request_text": text,
        "predicted_category": str(predictions[0]),
        "confidence": float(confidence[0])
    }
if __name__ == "__main__":
    app.run(debug=True)
