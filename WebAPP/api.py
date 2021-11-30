"""
Note this file contains NO flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import dill
import numpy as np
import pandas as pd
import re
import os


#--------------------prediction------------------------
# Open relevant models
with open(os.path.abspath("static/models/XGBoost_Model.joblib"), "rb") as to_read:
    model = dill.load(to_read)

with open(os.path.abspath("static/models/District_Transformer.pkl"), "rb") as to_read:
     transformer = pickle.load(to_read)

#read main file        
main_df = pd.read_csv('static/database/main_df.csv')
main_df.loc[main_df.district == 'Changi Airport / Changi Village (D17)', 'mrt_name'] = 'Tampines East Mrt'
main_df.dropna(subset=['mrt_name'], inplace=True)

def convert(x_input):
    """
    Function makes sure the features are fed to the model in the same order the
    model expects them.
    #['sqft', 'built_year', 'walking_time', 'district_number', 'mrt_name_Dhoby Ghaut MRT', 'mrt_name_Holland Village MRT']
    
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}
    
    Output:
    DataFrame
    """
    
    #convert district to encoded district number
    mapping = transformer.encoder_dict_['district_number']
    
    #regex to extract district number
    global district_number_before_transform
    
    district_number_before_transform = int(re.findall(f'D([\d]+)',  x_input['district'])[0])
    district_number = mapping[district_number_before_transform]
    
    
    
    Dhoby_Ghaut_MRT = 0
    Holland_Village_MRT = 0
    
    if x_input['mrt'] == 'Dhoby Ghaut MRT':
        Dhoby_Ghaut_MRT = 1
    
    if x_input['mrt'] == 'Holland Village MRT':
        Holland_Village_MRT = 1    
      
    model_input = np.array([
        int(x_input['sqft']),
        int(x_input['built_year']),
        int(x_input['walking_time_to_mrt']),
        district_number,
        Dhoby_Ghaut_MRT,
        Holland_Village_MRT
        
    ])
    
    return np.reshape(model_input, (-1, 6))


def make_prediction(x_input):
    
    """
    Predict Rental Price
    """
    
    model_input = convert(x_input)
    
    return int(model.predict(model_input))
    
district_list = main_df.district.unique().tolist()
mrt_list = main_df.mrt_name.unique().tolist()

year_min = main_df.built_year.min()
year_max = main_df.built_year.max()

district_mrt = dict()
for district_key in district_list:
    value = list(set(main_df.loc[main_df.district == district_key, 'mrt_name'].values))
    district_mrt[district_key] = value

# for import
district_list

# for import
district_mrt



#----------------------------recommendation-----------------------
with open(os.path.abspath("static/models/Kneighbour.pkl"), "rb") as to_load:
    neigh = pickle.load(to_load)

with open(os.path.abspath("static/database/df_recommender.pkl"), "rb") as to_load:
    df_recommender = pickle.load(to_load)   

with open(os.path.abspath("static/models/onehotencoder_Transformer.pkl"), "rb") as to_load:
    ct = pickle.load(to_load)

with open(os.path.abspath("static/models/FeatureImportanceScale.joblib"), "rb") as to_load:
    feature_importance = dill.load(to_load)

with open(os.path.abspath("static/models/KneighbourCluster.pkl"), "rb") as to_load:
    neigh_clus = pickle.load(to_load)

with open(os.path.abspath("static/models/FeatureImportanceScaleCluster.joblib"), "rb") as to_load:
    feature_importance_cluster = dill.load(to_load)

debug = False

def customtransformation(X):
    """
    Function to combine
    
    1.OneHotEncoder
    2.MixMaxScaler
    3.FeatureWeighting
    """
    #combine feature_name
    #feature of OHE
    feature1 = ct.named_transformers_.onehotencoder.get_feature_names().tolist()
    
    #remaining feature Note the feature order is important as this is the results of OHE transformation
    feature2 = ['built_year', 'walking_time_to_mrt', 'sqft', 'price_month', 'district', 'cluster_label', 'pool', 'gym', ]

    #combine 
    all_feature = feature1 + feature2
    
    X = pd.DataFrame.from_dict(X, orient='index').T
    
    #rearrange columns as per fit
    rearrange = ['district', 'mrt', 'built_year', 'walking_time_to_mrt', 'sqft', 'pool', 'gym', 'price_month', 'cluster_label']

    X = X[rearrange]
    
    # #casting numerical value

    for col in X:
        X[col] = pd.to_numeric(X[col], errors='ignore')
    #transform to Dataframe for Feature Importance    
    X_tr = pd.DataFrame(ct.transform(X).toarray(), columns=all_feature)
        
    return X_tr

def recommendlisting(user_input, prediction):
    user_input_sim = user_input.copy()

    district_number_before_transform = int(re.findall(f'D([\d]+)',  user_input_sim['district'])[0])

    user_input_sim['district'] = district_number_before_transform
    user_input_sim['price_month'] = prediction

    X = customtransformation(user_input_sim)
    X = feature_importance.transform(X)
    
    X = X.drop('cluster_label', axis=1)
    
    index = neigh.kneighbors(X,  return_distance=False)
    selected_index = np.reshape(index, -1)
    
    sim = df_recommender.iloc[selected_index]
    
    if debug == True:
        return sim

    price_month = sim.price_month.tolist()
    district = sim.district.tolist()
    detailed_address = sim.detailed_address.tolist()
    bedrooms = sim.bedrooms.tolist()
    bathrooms = sim.bathrooms.tolist()
    sqft = sim.sqft.tolist()
    built_year = sim.built_year.tolist()
    mrt = sim.mrt.tolist()
    walking_time_to_mrt = sim.walking_time_to_mrt.tolist()
    pool = sim.pool.tolist()
    gym = sim.gym.tolist()
    link = sim.link.tolist()
    picture_url	= sim.picture_url.tolist()
    
    return (price_month, 
        district, 
        detailed_address, 
        bedrooms, 
        bathrooms, 
        sqft, 
        built_year, 
        mrt, 
        walking_time_to_mrt,
        pool,
        gym, 
        link, 
        picture_url)

def recommendneighbhour(user_input, prediction):
    user_input_neighbhour = user_input.copy()
    district_number_before_transform = int(re.findall(f'D([\d]+)',  user_input_neighbhour['district'])[0])
    user_input_neighbhour['district'] = district_number_before_transform
    user_input_neighbhour['price_month'] = prediction   

    X = customtransformation(user_input_neighbhour)
    X = feature_importance_cluster.transform(X)
    
    X = X.drop('district', axis=1)
    
    index = neigh_clus.kneighbors(X,  return_distance=False)
    selected_index = np.reshape(index, -1)
    
    sim = df_recommender.iloc[selected_index]
    
    if debug == True:
        return sim    
    
    price_month = sim.price_month.tolist()
    district = sim.district.tolist()
    detailed_address = sim.detailed_address.tolist()
    bedrooms = sim.bedrooms.tolist()
    bathrooms = sim.bathrooms.tolist()
    sqft = sim.sqft.tolist()
    built_year = sim.built_year.tolist()
    mrt = sim.mrt.tolist()
    walking_time_to_mrt = sim.walking_time_to_mrt.tolist()
    pool = sim.pool.tolist()
    gym = sim.gym.tolist()
    link = sim.link.tolist()
    picture_url	= sim.picture_url.tolist()
    
    return (price_month, 
        district, 
        detailed_address, 
        bedrooms, 
        bathrooms, 
        sqft, 
        built_year, 
        mrt, 
        walking_time_to_mrt,
        pool,
        gym, 
        link, 
        picture_url)

if __name__ == '__main__':
    user_input = {
        'district' : 'Beach Road / Bugis / Rochor (D7)',
        'mrt' : 'Tampines MRT',
        'built_year' : '1980',
        'walking_time_to_mrt' : '22',
        'sqft' : '123',
        'pool' : '1',
        'gym': '1',
        'cluster_label' : 2
    }

    debug = True

    prediction = make_prediction(user_input)
    print(prediction)
    listing = recommendlisting(user_input, prediction)
    print(listing)
    neighbhourhood = recommendneighbhour(user_input, prediction)
    print(neighbhourhood)

    
