from flask import Flask, render_template, request
from api import district_list, district_mrt, mrt_list, year_min, year_max, make_prediction, recommendlisting
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
    user_input = request.args.to_dict()
    
    prediction = make_prediction(user_input)

    recommendation = recommendlisting(user_input, prediction)

    price_month = recommendation[0]
    district = recommendation[1]
    detailed_address = recommendation[2]
    bedrooms = recommendation[3]
    bathrooms = recommendation[4]
    sqft = recommendation[5]
    built_year = recommendation[6]
    mrt = recommendation[7]
    walking_time_to_mrt = recommendation[8]
    link = recommendation[9]
    picture = recommendation[10]
    
    return render_template('results.html', 
                            prediction = prediction,
                            price_month = price_month,
                            district = district,
                            detailed_address = detailed_address,
                            bedrooms = bedrooms,
                            bathrooms = bathrooms,
                            sqft = sqft,
                            built_year = built_year,
                            mrt = mrt,
                            walking_time_to_mrt = walking_time_to_mrt,
                            link = link,
                            picture = picture)

if __name__ == "__main__":
    app.run(debug=True)
