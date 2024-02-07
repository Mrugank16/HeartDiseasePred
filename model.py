import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('data.csv')

def encode(value):
  if value=='Presence':
    return 1
  elif value=='Absence':
    return 0

dataset['Heart Disease'] = dataset['Heart Disease'].apply(encode)


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
     


from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[55,0,2,130,415,0,1,110,0,1.5,1,2,4]]))