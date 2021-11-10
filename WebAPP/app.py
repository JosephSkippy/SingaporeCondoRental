from flask import Flask, render_template, request
from api import district_list, district_mrt, mrt_list, year_min, year_max, make_prediction
import numpy as np
import pandas as pd
import pickle
import json

app = Flask(__name__)
district_mrt_json = json.dumps(district_mrt)


@app.route('/')
@app.route("/predict", methods=["POST", "GET"])
def form():
    # use flask's render_template function to display an html page
    return render_template('base.html',
                            district_list = district_list,
                            district_mrt = district_mrt_json,
                            mrt_list = mrt_list,
                            year_min = int(year_min),
                            year_max = int(year_max)
                           
     )

@app.route('/results')
def predict():
    # use flask's render_template function to display an html page
    user_input = request.args
    
    prediction = make_prediction(user_input)
    
    return render_template('results.html', prediction = prediction)


if __name__ == "__main__":
    app.run(debug=True)
