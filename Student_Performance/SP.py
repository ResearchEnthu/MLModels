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

columns = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Performance Index"
]

# Loop through each column to plot its distribution
# for col in columns:
#     plt.figure(figsize=(8, 4))  # Set the figure size
#     sns.histplot(data[col], kde=True, bins=30, color="skyblue")  # Histogram with KDE curve
#     plt.title(f'Distribution of {col}')  # Set the plot title
#     plt.xlabel(col)  # Set x-axis label
#     plt.ylabel('Frequency')  # Set y-axis label
#     plt.grid(True)  # Add grid for better readability
#     plt.show()  # Show the plot
# clr = data.corr(numeric_only=True)
# sns.heatmap(clr, annot=True, cmap='Greens')
# plt.title("Correlation between Features")
# plt.show()

# sns.pairplot(data)
# plt.show()

data.drop(columns=['Extracurricular Activities','Sample Question Papers Practiced', 'Sleep Hours'], inplace=True)
data.head()

# X = data.drop(columns=["Performance Index"]).values
# y = data["Performance Index"].values.reshape(-1, 1)

cols = list(data.columns)
cols
x = data[cols[0:-1]]
y = data[cols[-1]]



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# from sklearn.linear_model import LinearRegression
# Model = LinearRegression()
# Model.fit(X_train,y_train)
# prediction = Model.predict(X_test)
# print("Prediction value:", prediction)


# from sklearn.metrics import r2_score
# r2_score(y_test, prediction)
# score = r2_score
# print("r2_score calculation:", score)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 23)
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    predict = cross_val_score(model, x, y, cv=10)
    return predict.mean()

def model_metrics(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def print_model_metrics(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('R2 Square', r2_square)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

coeffs = pd.DataFrame(data=[lr.coef_], columns=[x.columns])
coeffs.insert(0, "Model", "Linear Regression")
print("Coefficient value:", coeffs)

pred = lr.predict(x_test)
print("Prediction value:",pred)
pd.DataFrame({'True' : y_test,
             'Predicted' : pred}).plot.scatter(x='True', y='Predicted')
pd.DataFrame({'Error values': (y_test - pred)}).plot.kde()

test_pred = lr.predict(x_test)
train_pred = lr.predict(x_train)

print('Test set evaluation:')
print_model_metrics(y_test, test_pred)
print('\n')
print('Train set evaluation:')
print_model_metrics(y_train, train_pred)

score = lr.score(x_test, y_test)
print("R² score:", score)


param_grid = {
    'n_estimators' : [100,200,300],
    'max_depth' : [None,5,10]
}

rfr = RandomForestRegressor(random_state = 23)

grid_search = GridSearchCV(rfr,param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train,y_train)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score (MSE):", best_score)

import pickle

with open("classifier.pkl", "wb") as model_file:
    pickle.dump(lr, model_file)