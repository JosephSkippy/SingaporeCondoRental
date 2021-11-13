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

#--------------------prediction------------------------
# Open relevant models
with open("static\models\XGBoost_Model.pkl", "rb") as to_read:
    model = pickle.load(to_read)

with open("static\models\District_Transformer.pkl", "rb") as to_read:
     transformer = pickle.load(to_read)

#read main file        
main_df = pd.read_csv('..\Data\main_df.csv')
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
with open("static\models\Kneighbour.pkl", "rb") as to_load:
    neigh = pickle.load(to_load)

with open("static\database\df_recommender.pkl", "rb") as to_load:
    df_recommender = pickle.load(to_load)   

with open("static\models\onehotencoder_Transformer.pkl", "rb") as to_load:
    ct = pickle.load(to_load)

with open("static\models\FeatureImportanceScale.joblib", "rb") as to_load:
    feature_importance = dill.load(to_load)
       
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
    
    #remaining feature
    feature2 = ['built_year', 'walking_time_to_mrt', 'sqft', 'price_month', 'district', 'pool', 'gym']

    #combine 
    all_feature = feature1 + feature2
    
    X = pd.DataFrame.from_dict(X, orient='index').T
    
    #rearrange columns as per fit
    rearrange = ['district', 'mrt', 'built_year', 'walking_time_to_mrt', 'sqft', 'pool', 'gym', 'price_month']

    X = X[rearrange]
    
    # #casting numerical value

    for col in X:
        X[col] = pd.to_numeric(X[col], errors='ignore')
    #transform to Dataframe for Feature Importance    
    X_tr = pd.DataFrame(ct.transform(X).toarray(), columns=all_feature)
    X_tr = feature_importance.transform(X_tr)
    
    return X_tr

def recommendlisting(user_input, prediction):

    district_number_before_transform = int(re.findall(f'D([\d]+)',  user_input['district'])[0])

    user_input['district'] = district_number_before_transform
    user_input['price_month'] = prediction
       
    X = customtransformation(user_input)
    
    index = neigh.kneighbors(X,  return_distance=False)
    selected_index = np.reshape(index, -1)
    
    sim = df_recommender.iloc[selected_index]
    
    price_month = sim.price_month.tolist()
    district = sim.district.tolist()
    detailed_address = sim.detailed_address.tolist()
    bedrooms = sim.bedrooms.tolist()
    bathrooms = sim.bathrooms.tolist()
    sqft = sim.sqft.tolist()
    built_year = sim.built_year.tolist()
    mrt = sim.mrt.tolist()
    walking_time_to_mrt = sim.walking_time_to_mrt.tolist()
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
        'gym': '1'
    }

    prediction = make_prediction(user_input)
    print(prediction)
    listing = recommendlisting(user_input, prediction)

    
