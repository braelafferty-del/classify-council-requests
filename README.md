# Council Service Requests

A project that uses TF-IDF and Logistic Regression to classify council service requests into categories.

## Features
- Cleans requests to text and removes duplicates
- Splits the dataset to a 80/20 training and testing split
- Converts text into numerical features using TF-IDF
- Classifies requests as road, waste, water and safety.
- Saves the model
- Has a flask API with /health /categories /predict
- /predict catcher if request_text is missing or has invalid input.
## Setup
```
git clone https://github.com/braelafferty-del/council-service-requests.git
cd council-service-requests
python3 -m pip install -r requirements.txt
```
### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check that the API is running |
| GET | `/categories` | List the model’s categories |
| POST | `/predict` | Predict a category and return confidence |

## Usage

### Start the API

start Flask:

```
python3 api/app.py
```
### Send a JSON object to `/predict`:

```json
{
  "request_text": "Missed rubbish collection outside my house"
}
```

### Prepare Data

Run the preparation script:

```
python3 notebooks_or_scripts/prepare_data.py
```

This cleans the request text, removes duplicates and saves the cleaned dataset and the 80/20 training/testing split in `data/processed`.

### Train Model

```
python3 notebooks_or_scripts/train_model.py
```

This trains the classifier, prints accuracy and a classification report based on the data given and saves the model, vectoriser and example predictions.




