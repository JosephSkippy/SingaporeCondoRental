from flask import Flask, render_template, request
from api import district_list
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
def form():
    # use flask's render_template function to display an html page
    return render_template('base.html',
                            district_list = district_list
     )

@app.route('/results')
def predict():
    # use flask's render_template function to display an html page
    user_input = request.args
    data = [
        user_input['OverallQual'],
        user_input['FullBath'],
        user_input['GarageArea'],
        user_input['LotArea'],
    ]
    X = np.reshape(data, (1, -1))
    prediction = ols_model.predict(X)
    return render_template('results.html', prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)
