# Flask API 

from flask import Flask

app = Flask(__name__)
# Load model and vectoriser 
model = joblib.load("../model/model.joblib")
vectoriser = joblib.load("../model/vectoriser.joblib")

# /Health route to show the app is working
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
