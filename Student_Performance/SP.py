import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv(r"C:\Users\alokk\OneDrive\Documents\Research_POC\MLModels\Student_Performance.csv")
print("Student performance data:",data)

data.head(20)
data.info()
data.describe()

print("\nNull values in each column:",data.isnull().sum())
print("shape of data:",data.shape)  
print("describe the data set",data.describe())
print("shape of data:",data.shape)

data['Extracurricular Activities'].replace(['Yes','No'],[True,False],inplace=True)
print("Extracurricular activities", data['Extracurricular Activities'])

import plotly.express as px
fig=px.imshow(data.corr(),
             text_auto=True,
             color_continuous_scale='RdBu')
fig.show()

from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split as tts
model=lm.LinearRegression()

y=data['Performance Index']
X=data.drop('Performance Index',axis=1)

X_train,X_test,y_train,y_test=tts(X,y,test_size=0.15,random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Predictkion:", y_pred)

from sklearn.metrics import mean_squared_error,r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))

import plotly.express as px
import pandas as pd

plot_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

scatter_plot = px.scatter(
    plot_df,  
    x='Actual',  
    y=plot_df.index, 
    title="Actual vs Predicted",
    labels={"Actual": "Actual Values", "index": "Index"}, 
)


scatter_plot.add_scatter(
    x=plot_df['Predicted'], 
    y=plot_df.index,  
    mode='markers',  
    hovertemplate="Predicted: %{x}<br>Index: %{y}",  
    marker=dict(opacity=0.5),  
    name="Predicted" 
)
