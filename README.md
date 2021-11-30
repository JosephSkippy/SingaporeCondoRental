#  Smarter Condominium Rental Search
![alt text](https://www.ura.gov.sg/-/media/Corporate/Planning/Master-Plan/Housing/Housing_3.jpg?h=400&la=en&w=640)

# Problem Statement
Foreigners who seek to rent a condominium unit in Singapore may not be aware of the market price.

In this project we will create a webapp to predict the monthly rental price for a condominium and recommend users the best values, similar units and neighbhourhoods for the users.

# Executive Summary

**For Predicting Monthly Rental**
<br>With over 7000 thousands of listings scrapped from property website,  which factors were most predictive of condominium monthly rental have been determined, regression model(XGBoost) was built and deployed via a Flask app on [Heroku](https://smartercondosearch.herokuapp.com/predict)

Following are the features that best predict condominium monthly rental:-

1. Square Feet
2. District
3. MRT
4. MRT Walking Time
5. Year Built

The model has achieved 0.92 adjusted r2 score on out of sample data. Learning curve has implies that increasing training sample can further increase the model accuracy.


**For Recommending Neighbourhood & Similar Listing**

Following features have been retrieved via Foursquare API for a given listing to determine the types of neighbourhood.

1. Arts & Entertainment
2. Food
3. Shop & Services
4. College & University
5. Night Life Spot
6. Outdoor and Recreation

Refer to [Foursquare documentation](https://developer.foursquare.com/docs/build-with-foursquare/categories/) for details of the above category

With **UMAP** for dimension reduction & **HDBscan clustering**, 5 distinct neighbhourhoods have been found.

![GitHub Dark](picture/cluster.png#gh-dark-mode-only)


# Files

[00A MainData Acquisition.ipynb](00A_MainData_Acquisition.ipynb)
Scrapping of listing information via **Selenium**.

[00B_Data_Acquisition(LatLong).ipynb](00B_Data_Acquisition(LatLong).ipynb)
Retrieval of Lat Long information via **Geocoder API**

[00C_Creating_Singapore_District_Geojson.ipynb](00C_Creating_Singapore_District_Geojson.ipynb)
Creation of Singapore District Polygon Geometry

[00D_Data_Acquisiton_Neighbourhood.ipynb](00D_Data_Acquisiton_Neighbourhood.ipynb)
Retrieval of Neighbhourhood Information via **Foursquare API**

[01_Data_Cleaning.ipynb](01_Data_Cleaning.ipynb)
shows the process of cleaning data, handling outliers, missing values.

[02_EDA.ipynb](02_EDA.ipynb)
Univariate analysis of features, eg Distribution plots to understand the characteristic of our data.

Multivariate analysis of features and target to understand the correlation and predictive power of a feature.

# Methodology

### For Predicting Monthly Rental
**Data collection**
<br> Webscraping via Selenium web on www.99co.com for condominium listing

**Data cleaning**
1. Remove duplicate row
2. Remove feature with high missing values > 30%
3. Remove row-wise listing with missing value ( less than 10% of total row)
4. Imputation of missing values via (Arbitrary & Random Forest)
5. Outlier (Anomaly) Removal

**EDA**
1. [District Insight](picture/Analysis_map.html)
2. Multi-collinearity via Clustering & Correlation Heatmap

**Pre-Processing**
1. CountVectorizer
2. RareLabelEncoding & OneHotEncoding
3. Ordinal Encoding
4. Box-cox transformation  for Linear Regression

**Model Selection**

**Validation Score**
- | #rmse_mean | #rmse_std | #adjusted_r2_mean | #adjusted_r2_std
--- | --- | --- | --- |--- |
LinearRegression | 3127 | 279 | 0.5 | 0.02 |
SVM Regression | 2110 | 383 | 0.77 | 0.06 |
XGBoost Regression | 1445 | 329 | 0.88 | 0.04 |

**Test Score**
<br>Final_XGB_test_rmse: 1298
<br>Final_XGB_test_adjusted_r2: 0.90

### For Recommending Neighbhourhood
**Data collection**
<br> Webscraping via Foursquare API & BeautifulSoup

 **Pre-Processing**
 <br>UMAP

![UMAP](picture/UMAP.png)
Cluster with neighbors=12, min_dist=0 were selected as it retains the local cluster structure and global structure

 **Clustering**
 <br>HDBscan
![HDBscan](picture/HDBscan.png)
Selection of hyperparameter based on visual


 **Recommendation**

 Content based recommendation based on k-neighbhours algorithm.

# Conclusion
The production model is a XGBoost Regression model. Given its relatively high adjusted_r2 (0.91) on the test set, we can conclude that the model generalises well on unseen data.


Despite the model's relatively high performance, there is still room for improvement as shown in the learning curve that getting more data will improve the accuracy.
