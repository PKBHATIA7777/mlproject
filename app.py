# ─────────────────────────────────────────────────────────────
# Flask is a lightweight web framework for Python.
# It lets us create a website/web app using Python code.
# 'request'       → used to read data sent from the HTML form
# 'render_template' → used to send/show an HTML page to the user
# ─────────────────────────────────────────────────────────────
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# We import our two custom classes from the predict pipeline file:
# CustomData      → a blueprint to hold the form data entered by user
# PredictPipeline → does the actual prediction using our saved model
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# ─────────────────────────────────────────────────────────────
# Create the Flask app.
# __name__ tells Flask where to find the templates folder etc.
# We give it two names (application & app) — both refer to same app.
# AWS Elastic Beanstalk looks for a variable named "application"
# so that is why we name it "application" and then alias it to "app".
# ─────────────────────────────────────────────────────────────
application = Flask(__name__)
app = application


# ─────────────────────────────────────────────────────────────
# ROUTE 1: Home Page  →  url: "/"
# When someone visits the home page, just show the index.html file.
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────
# ROUTE 2: Prediction Page  →  url: "/predictdata"
# This route handles TWO things:
#   GET  → User just visits the page → show blank form (home.html)
#   POST → User fills form and clicks "Predict" → run model → show result
# ─────────────────────────────────────────────────────────────
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        # User just opened the page — show the empty form
        return render_template('home.html')

    else:
        # ── STEP 1: Collect form data ────────────────────────────────
        # request.form.get('field_name') reads the value the user typed
        # in the HTML form for that specific input field.
        # We pack all these values into our CustomData object (like a box).
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # ── STEP 2: Convert to DataFrame ─────────────────────────────
        # Our ML model was trained on a pandas DataFrame.
        # So we convert the user's input into a DataFrame (like a table
        # with 1 row) so the model can understand it.
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Just for debugging — prints input in terminal

        # ── STEP 3: Run the Prediction ───────────────────────────────
        # PredictPipeline loads our saved model and preprocessor,
        # scales/encodes the input, and returns the predicted score.
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # ── STEP 4: Show the Result ──────────────────────────────────
        # We send the result back to home.html.
        # 'results=results[0]' passes the predicted score to the HTML page
        # so it can display it to the user.
        return render_template('home.html', results=results[0])


# ─────────────────────────────────────────────────────────────
# Start the Flask development server.
# host="0.0.0.0" → makes it accessible on your local network
# debug=True      → auto-reloads when you change code
# port=5001       → runs on http://localhost:5001
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)