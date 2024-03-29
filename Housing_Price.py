import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer #scikit learn : libraries for ML models
                                          #Imputer class: take care of missing data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
##import datasets

train_set=pd.read_csv("train_housing.csv")
test_set=pd.read_csv("test_housing.csv")


#train_set=pd.DataFrame(train_set)
test_set=test_set
## separate dependent and independent variable test set
X_test=test_set

## separate dependent and independent variable train set
Y_train=train_set.iloc[:, 80].values

X_train=train_set.iloc[:,:-1] #X_train = train_set-1

# clean training set

X_train['SaleType'].replace({'WD' : int(1), 'New': int(2), 'COD' : int(3), 'ConLI' : int(4), 'ConLD' : int(5), 'Oth' : int(6), 'ConLw': int(7)}, inplace=True)
X_train['SaleCondition'].replace({'Normal':1,'Abnorml':2,'Partial':3,'Alloca':4,'Family':5},inplace=True)

X_train=X_train.convert_objects(convert_numeric=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#str_cols=X_train.select_dtypes(exclude=numerics)
X_train=X_train.select_dtypes(include=numerics)


#clean test set

X_test['SaleType'].replace({'WD' : int(1), 'New': int(2), 'COD' : int(3), 'ConLI' : int(4), 'ConLD' : int(5), 'Oth' : int(6), 'ConLw': int(7)}, inplace=True)
X_test['SaleCondition'].replace({'Normal':1,'Abnorml':2,'Partial':3,'Alloca':4,'Family':5},inplace=True)

X_test=X_test.convert_objects(convert_numeric=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#str_cols=X_train.select_dtypes(exclude=numerics)
X_test=X_test.select_dtypes(include=numerics)




print(X_test)
## Impute values for training set

#Taking care of missing data
                   
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #recognises missing values
imputer = imputer.fit(X_train) 
X_train= imputer.transform(X_train)


#Impute values for test data
#Taking care of missing data
                   
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #recognises missing values
imputer = imputer.fit(X_train) 
X_test= imputer.transform(X_test)
print(X_test)
# fitting random forest regressor

regressor=RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, Y_train)
Y_pred=regressor.predict(X_test)
pd.DataFrame(Y_pred).to_csv('prediction.csv') #save locally for reference
#visualisation

import matplotlib.pyplot as plt

plt.xlabel('X_test', fontsize=5)
plt.ylabel('Y_pred', fontsize=5)
plt.xticks(X_test[:,37], Y_pred, fontsize=5, rotation=30)
plt.title('Housing Price Prediction - 2019')
plt.show()