import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"Dataset/avocado.csv")
print("Data load from dataset folder:",df)
print("Data information:",df.info())

df.drop(['Unnamed: 0', 'Date'],axis=1, inplace=True)
df.drop(['XLarge Bags'], axis=1, inplace=True)

print("Data load from dataset folder after Drop columns:",df)
print("dataset Descripttion",df.describe())

plt.hist(df['AveragePrice'], facecolor='violet', edgecolor='black',bins=10)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x=df['region'], y=df['AveragePrice'])
plt.xticks(rotation=80)
plt.show()

plt.hist(df['year'])
plt.show()

print("Check null value:",df.isnull().sum())

print("Check unique value:",df.type.unique())
print("Check unique value:",df.region.unique())

df1=pd.get_dummies(df['type'],drop_first=True)
df2=pd.get_dummies(df['region'],drop_first=True)
df3=pd.concat([df,df1,df2],axis=1)
print("Data Frame new after concatination",df3.head())

df3.drop(['type','region'],axis=1,inplace=True)

import sklearn
from sklearn.model_selection import train_test_split

X=df3.loc[:, 'Total Volume':'WestTexNewMexico']
y=df3[['AveragePrice']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import xgboost as xgb
from sklearn.metrics import r2_score
xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)

model=xgb.predict(X_test)
r2_score(y_test,model)
print("r2 score:,",r2_score(y_test,model))

from sklearn.ensemble import RandomForestRegressor
Rand_Reg= RandomForestRegressor(random_state=1)
Rand_Reg.fit(X_train,y_train)

RR_model=Rand_Reg.predict(X_test)
r2_score(y_test,RR_model)
print("r2 score Random regression:,",r2_score(y_test,RR_model))
