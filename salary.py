import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def salary_prediction(years):    
    path="C:\\Users\\ankit\\_Demo\\"
    data=path+'Salary_Data.csv'
    dataset=pd.read_csv(data)

    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values

    imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(x[:,:])
    x[:,:]=imputer.transform(x[:,:])

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

    regressor=LinearRegression()
    regressor.fit(x_train,y_train)

    x_test=np.array(years)
    x_test=x_test.reshape((1,-1))

    return regressor.predict(x_test)[0]



