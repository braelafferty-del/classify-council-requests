# Flask API 

from flask import Flask

app = Flask(__name__)

# /Health route to show the app is working
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
