# Council Service Requests

A project that uses TF-IDF and Logistic Regression to classify council service requests into the road, waste, water and safety categories.

## Requirements

- Python 3.13( May work on lower versions but tested on 3.13)

## Setup

Clone the repository and install the required packages:

```
git clone https://github.com/braelafferty-del/council-service-requests.git
cd council-service-requests
python3 -m pip install -r requirements.txt
```

## Usage

### Prepare Data

Run the preparation script:

```
python3 notebooks_or_scripts/prepare_data.py
```

This cleans the request text, removes duplicates and saves the cleaned dataset and training/testing split in `data/processed`.

### Train Model

```
python3 notebooks_or_scripts/train_model.py
```

This trains the classifier, prints accuracy and a classification report, and saves the model, vectoriser and example predictions.

### Start the API

start Flask:

```
python3 api/app.py
```

Keep this terminal running while testing the API.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check that the API is running |
| GET | `/categories` | List the model’s categories |
| POST | `/predict` | Predict a category and return confidence |

Send a JSON object to `/predict`:

```json
{
  "request_text": "Missed rubbish collection outside my house"
}
```

### Test a Prediction

Open a PowerShell terminal and run:

```
'{"request_text":"Missed rubbish collection outside my house"}' | curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@-"
```
