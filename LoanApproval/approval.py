import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


df = pd.read_csv(r"C:\Users\alokk\OneDrive\Documents\Research_POC\MLModels\Dataset\loan_approval_dataset.csv")
print("Data loading:",df)

df.head(20)
print("Data head:", df.head(20))
print("Data head:", df.tail(20))
print("Show Data Information:",df.info())
print("Show data Shap:",df.shape)

# Data cleaning
df.isna().sum()
print("Check NA:",df.isna().sum())

print("Check name of all columns:",df.columns)

# Data set distribution
print("Dataset distribution:",df.describe())
print("Dataset distribution with duplication:",df.duplicated().sum())

# data visualization
numerical_cols = df.select_dtypes(include=np.number)

# get the corelation matrics
corre = numerical_cols.corr()
print("corelation map:", corre)

#ploting a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corre, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation HeatMap")
plt.show()

staus_counts = df[' loan_status'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(staus_counts, labels=staus_counts.index, autopct='%1.1f%%',startangle=90, colors=['green','red'])
plt.title("Loan Approval vs Loan Rejection")
plt.show()

sns.pairplot(df[[' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' loan_status']], hue=' loan_status')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

df = df.drop(['loan_id',' no_of_dependents',' education'],axis = 1)
print("Drop the columns:", df)

print("Drop the na:",df.dropna(inplace=True))

X = df.drop(' loan_status', axis=1)
y = df[" loan_status"]



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.dtypes)
print(X_train)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Logistic regression

model = LogisticRegression()
model.fit(X_train_encoded, y_train)
y_prediction = model.predict(X_test_encoded)
print("Logistic regression Accuracy:",accuracy_score(y_test,y_prediction))


# Decesion tree model
dt_model = DecisionTreeClassifier(random_state = 42)
dt_model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
print('Decision Tree Accuracy:',accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))

# Random Forest model 

rf_model = RandomForestClassifier()
rf_model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
print('Random Forest Accuracy:',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Support Vector Machine model

model = SVC(kernel = 'rbf')
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
print('SVM Accuracy:',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Define and evaluate models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    results[name] = accuracy_score(y_test, y_pred)

for model, acc in results.items():
    print(f"{model}: {acc:.4f}")


# Example using best model
best_model = models["Decision Tree"]
y_pred = best_model.predict(X_test_encoded)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# Function to plot confusion matrix and print report
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test_encoded)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(f"Classification Report - {name}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Evaluate Decision Tree
evaluate_model("Decision Tree", models["Decision Tree"], X_test_encoded, y_test)

# Evaluate Random Forest
evaluate_model("Random Forest", models["Random Forest"], X_test_encoded, y_test)
