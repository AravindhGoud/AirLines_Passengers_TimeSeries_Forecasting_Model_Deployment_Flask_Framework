
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

model = pickle.load(open(r'C:\Users\aravi\OneDrive\Desktop\new\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('sample.html')

# Create an endpoint for the user to input the month and year
@app.route('/predict', methods=['POST'])
def predict():
    
    num_months = int(request.form['month_year'])
    prediction = round(np.exp(model.forecast(num_months)))
    
    df = pd.DataFrame(prediction, columns=['Passengers'])
    df.index.name = 'Date'
    return render_template('results.html', prediction=df)


if __name__=="__main__":
    app.run(debug=True)
