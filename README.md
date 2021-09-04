## Ironhack Data Analytics Bootcamp 
# **Term Term Project - Predicting house prices**

## **Goal of the project**

The objective of this project is to build a machine learning model to predict the selling prices of houses based on a variety of features on which the value of the house is evaluated.


## **Exploring the Data**

The data used contains about 22.000 rows and consists of historic data of houses sold between May 2014 to May 2015

## **Project Content** 

The process was done in the following steps:

1.  Check general details of data (shape, data types, null values)
2.  Generalize headers names
3.  Clean specific columns(dealing with null values, clean content, remove outliers, change data types etc.)
4.  Divide into cateogorical, numerical data and prepare them for the model (check correlations, distributions, encode categorical data)
5.  Concat and apply transformations (BoxCox and Normalizer)
6.  Run 3 different regression models

## **Results: Predict housing prices**

1. Without transformer the R2 scores of the model resulted in: 

Linear Regression: 0.86, Mean absolute error: 67,470.00
KNN Regressor (n=7): 0.57, Mean absolute error: 124,525.00
MLP Regressor: R2 score: 0.58, Mean absolute error: 127,339.00

Linear Regression provided the best result. KNN and MLP are really low compared.

2. Applying Boxcox & Normalizer, the R2 score resulted in: tbd


## **Libraries used**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import Functions

---






