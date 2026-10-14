# classify-council-requests
Using machine learning to classify requests for the council using TF-IDF and LogisticRegression


# Instructions
git clone https://github.com/braelafferty/classify-council-requests.git
cd classify-council-requests
pip3 install -r requirements.txt
# Train model
cd notebooks_or_scripts
python3 prepare_data.py
python3 train_model.py

# Run flask app 
python3 api/app.py
