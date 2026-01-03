import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

#1.load the dataset
housing=pd.read_csv("/Users/sankalpmakol/Desktop/ML-Projects/project1/housing.csv")

#2.create stratified test set
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)
    
housing = strat_train_set.copy()

#3.predicting features and labels
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

print(housing,housing_labels)

#4.List the num and categoraical attributes
num_attributes = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

#5.making pipelines

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

#6.transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)