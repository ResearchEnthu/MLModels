import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\alokk\OneDrive\Documents\Research_POC\MLModels\Dataset\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
print("Load the dataset:", df)
print("dataset upto 20 records:",df.head(20))

# Exploring the data
print("Exploring the datasets:", df.info())
print("Exploring the datasets:", df.describe())
print("Exploring the datasets:", df.duplicated().sum())

# Data Cleaning 
print("Drop unnecessary features:",df.drop(['Income', 'Education'], axis = 1, inplace = True))
print("Load the dataset after droping some feature:", df)
df = df.drop_duplicates()
print(df.duplicated().sum())

# Converting datatypes of some features

df['Diabetes_binary']=df['Diabetes_binary'].astype(int)
df['Age']=df['Age'].astype(int)
df['Sex']=df['Sex'].astype(int)
df['Smoker']=df['Smoker'].astype(int)
df['Stroke']=df['Stroke'].astype(int)

# Data Visualization

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot 1: Distribution of 'Diabetes_binary'
sns.countplot(x='Diabetes_binary', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Diabetes_binary')
axes[0, 0].set_xlabel('Diabetes_binary')
axes[0, 0].set_ylabel('Count')

# Plot 2: Distribution of 'Smoker'
sns.countplot(x='Smoker', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Smokers')
axes[0, 1].set_xlabel('Smoker')
axes[0, 1].set_ylabel('Count')

# Plot 3: Distribution of 'HvyAlcoholConsump'
sns.countplot(x='HvyAlcoholConsump', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of HvyAlcoholConsump')
axes[1, 0].set_xlabel('HvyAlcoholConsump')
axes[1, 0].set_ylabel('Count')

# Plot 4: Distribution of 'Age'
sns.countplot(x='Age', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Age')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# data Preprocessing 

X = df[df.columns[1:]]
y = df['Diabetes_binary']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 50)

print("Train data of x:", x_train.shape)    
print("Train data of y:", y_train.shape)
print("Test data of x:", x_test.shape)
print("Test data of y:", y_test.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Building Logistic Regression Model

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(x_train_scaled, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_test = reg.predict(x_test_scaled)

reg_accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {round(reg_accuracy_test * 100,2)}%")

y_pred_train = reg.predict(x_train_scaled)

reg_accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy: {round(reg_accuracy_train * 100,2)}%")

print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8,5))

sns.heatmap(cm, annot=True, fmt ='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusing Matrix')
plt.show()


