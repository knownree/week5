import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cab=pd.read_csv('Cab.csv')
x=cab[['KM Travelled','Price Charged']]
y=cab['Cost of Trip']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=101)
lm=LinearRegression
lm.fit(x_train,y_train)
pickle.dump(lm,open('model.pkl','web'))