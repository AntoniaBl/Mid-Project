import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols


#Clean Data

#Replace Value in Columns
def replace_in_column(column,find,replace_by):
    for df1.rows in df1[column]:
        df1[column] = df1[column].str.replace(find,replace_by)

#Replace nall with mean in column
def fillup_mean(column):
    column_mean = df1[column].mean()
    df1[column] = df[column].fillna(column_mean)

#Give year
def yearinrows(df,column):
    for df.rows in df[column]:
        df[column] = df[column].year

#Convert data types

def convert_data_types(df,convertto, incolumns=[]):
    for column in incolumns:
        if convertto == "object":
            df[column]=df[column].astype('object')
        elif convertto == "date":
            df[column]=pd.to_datetime(df[column])
        elif convertto == "number":
            df[column]=pd.to_numeric(df[column])



#EDA 

#continous, discrete or categorical?

def sort_variables(data): 
    discrete = pd.DataFrame()
    continuous = pd.DataFrame()
    categorical = pd.DataFrame()
    for col in data:
        if data[col].dtypes == object:
            categorical[col] = data[col]
        elif len(data[col].unique()) >= 50:
            continuous[col] = data[col]
        else:
            discrete[col] = data[col]
    print("Disrete columns:",discrete.columns)
    print("Continuous Columns:",continuous.columns)
    print("Categorical Columns:",categorical.columns),

#Cap outliers
def cap_outliers(df, threshold=1.5, in_columns=[], skip_columns=[]): #default values need to be static, otherwise it does not work
    for column in in_columns:
        if column not in skip_columns:
            print ('Variables capped:',column)
            upper = np.percentile(df[column],75)
            lower = np.percentile(df[column],25)
            iqr = upper - lower
            upper_limit = upper + (threshold * iqr)
            lower_limit = lower - (threshold * iqr)
            df.loc[df[column] > upper_limit, column] = upper_limit
            df.loc[df[column] < lower_limit, column] = lower_limit
    return df

#Winsorize Outliers for one column
def winsorize_one(df, variable_name):    
    q1 = df[variable_name].quantile(0.25)
    q3 = df[variable_name].quantile(0.75)
    iqr = q3-q1
    outer_fence = 3*iqr
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    return outer_fence_le, outer_fence_ue

#Winsorize Outliers for more column
def winsorize_more(df, in_columns=[],skip_columns=[]): 
    for column in in_columns:
        if column not in skip_columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3-q1
            outer_fence = 3*iqr
            outer_fence_le = q1-outer_fence
            outer_fence_ue = q3+outer_fence
            return outer_fence_le, outer_fence_ue


#Show distributions
def distributions(df):
    for col in df.columns:
        sns.distplot(df[col])
        plt.show()

#Check distributions after Scaling Columns (Standard Scaler, MinMaxScaler, Normalization)
#For each method: create copy of df (data_copy), apply method, print distribution of columns

def Scalings(data, scaling_methods=[]):
    for method in scaling_methods:
        data_copy = data.copy()
        transformer = method
        transformer.fit(data_copy)
        x = transformer.transform(data_copy)
        numericals_final = pd.DataFrame(x, columns = data.columns)
        print("Distributions of:",method)
        distributions(numericals_final) 

#BoxCox Transformation
def boxcox_transform(data):
    _ci = {column: None for column in data.columns}
    for col in data:
        data[col] = np.where(data[col]<=0, np.NAN, data[col]) 
        data[col] = data[col].fillna(data[col].mean())
        transformed_data, ci = stats.boxcox(data[col])
        data[col] = transformed_data
        _ci[col] = [ci] 
    return data, _ci




#apply & check regression models
#[LinearRegression(), KNeighborsRegressor(n_neighbors=7)]
def apply_models (models=[]):
    for model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=18)
        model.fit(X_train, y_train)
        R2score = model.score(X_test, y_test)
        print(model,":",R2score)
