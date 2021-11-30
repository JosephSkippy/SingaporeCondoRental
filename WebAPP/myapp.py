from flask import Flask, render_template, request
from api import district_list, district_mrt, mrt_list, year_min, year_max, make_prediction, recommendlisting, recommendneighbhour
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

    #recommend similar listing
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
    pool = recommendation[9]
    gym = recommendation[10]
    link = recommendation[11]
    picture = recommendation[12]


    #recommend similar neighbhourhood
    recommendation_neigh = recommendneighbhour(user_input, prediction)

    price_month_neigh = recommendation_neigh[0]
    district_neigh = recommendation_neigh[1]
    detailed_address_neigh = recommendation_neigh[2]
    bedrooms_neigh = recommendation_neigh[3]
    bathrooms_neigh = recommendation_neigh[4]
    sqft_neigh = recommendation_neigh[5]
    built_year_neigh = recommendation_neigh[6]
    mrt_neigh = recommendation_neigh[7]
    walking_time_to_mrt_neigh = recommendation_neigh[8]
    pool_neigh = recommendation_neigh[9]
    gym_neigh = recommendation_neigh[10]
    link_neigh = recommendation_neigh[11]
    picture_neigh = recommendation_neigh[12]





    
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
                            pool = pool,
                            gym = gym,
                            link = link,
                            picture = picture,
                            

                            price_month_neigh = price_month_neigh,
                            district_neigh = district_neigh,
                            detailed_address_neigh = detailed_address_neigh,
                            bedrooms_neigh = bedrooms_neigh,
                            bathrooms_neigh = bathrooms_neigh,
                            sqft_neigh = sqft_neigh,
                            built_year_neigh = built_year_neigh,
                            mrt_neigh = mrt_neigh,
                            walking_time_to_mrt_neigh = walking_time_to_mrt_neigh,
                            pool_neigh = pool_neigh,
                            gym_neigh = gym_neigh,
                            link_neigh = link_neigh,
                            picture_neigh = picture_neigh
                            
                            )

@app.route('/about')
def about():

    return render_template('about.html')





if __name__ == "__main__":
    app.run(debug=True)
