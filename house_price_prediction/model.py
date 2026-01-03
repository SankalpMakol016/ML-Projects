import os
import pandas as pd
import numpy as np 
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score



MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribute,cat_attribute):
    #for numerical columns
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    #for cat columns
    cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    #constructing the full pipeline
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attributes),
        ("cat",cat_pipeline,cat_attributes)
    ])
    return full_pipeline
if not os.path.exists(MODEL_FILE):
    #lets train the model
    #load the dataset
    housing=pd.read_csv("/Users/sankalpmakol/Desktop/ML-Projects/project1/housing.csv")

    #create stratified test set
    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing,housing["income_cat"]):
        strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
        strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)
    housing = strat_train_set.copy()
    #storing test data as input.csv
    strat_test_set.to_csv("input.csv",index = False)
    
    #predicting features and labels
    housing_labels=housing["median_house_value"].copy()
    housing_features=housing.drop("median_house_value",axis=1)
    
    #List the num and categoraical attributes
    num_attributes = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]
    
    #making pipelines
    pipeline=build_pipeline(num_attributes,cat_attributes)
    #transform the data
    housing_prepared = pipeline.fit_transform(housing_features)
    #print(housing_prepared)
    
    #train the model 
    #Random Forest model
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("congrats model trained!!")
    
else:
    #inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions
    
    input_data.to_csv("output.csv",index=False)
    print("inference is complete results saved to output.csv")    