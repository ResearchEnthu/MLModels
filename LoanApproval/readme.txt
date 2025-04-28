#Data Problem Statement¶
# there are 13 different columns in the dataset,

# ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

# Out of these 13 columns, 1 column (Loan_Status) is the "target" or "label" or "y", meaning it is what we are trying to predict. The remaining 12 columns are considered as "inputs" or "features" or "X".

# It is a classification problem, because the target variable Loan_Status is categorical with values:

# Y – Loan approved

# N – Loan not approved

# The goal is to build a machine learning model that can accurately predict whether a loan will be approved or not, based on the applicant's details.


Check name of all columns: Index(['loan_id', ' no_of_dependents', ' education', ' self_employed',
       ' income_annum', ' loan_amount', ' loan_term', ' cibil_score',
       ' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value', ' loan_status'],


Machine Learning Model Training¶
Step 1 : Preparing X & y data

Step 2 : Spliting the data into train and test sets

Step 3 : Build an ML

Step 4 : Train the ML model with training data

Step 5 : Predict test data on the trained model

Step 6 : Calculate Mean Squared Error for further analysis


ValueError: could not convert string to float: ' No'
means that your feature data (X_train and X_test) still has string values (like ' No') in it, and LogisticRegression (and most scikit-learn models) expect only numeric input.

The specific issue ' No' (notice even the space before No) suggests your data is not clean — it contains string labels, possibly with extra whitespace.

To fix it: You need to preprocess your data and convert all categorical or string features into numeric form.

Example : from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

Report

 Loan Approval Classification In this project, we built a machine learning model to predict loan approval status based on various applicant features such as gender, marital status, education, employment, income, credit history, and loan amount.

 Data Preprocessing: We handled missing values, encoded categorical variables, and scaled numerical features where necessary to ensure model readiness.

Exploratory Data Analysis (EDA) revealed key insights:

Applicants with a credit history are significantly more likely to get their loans approved.

Higher applicant income and loan amount term slightly influence approval chances.

Model Building:

We tested various models such as Logistic Regression, Decision Trees, Random Forest.

Among these, Decision tree model performed the best with an accuracy of 92.9%. and Random forest model gives 88.2 % of accuracy.

Evaluation Metrics:

Accuracy, precision, recall, and F1-score were used to evaluate performance.

The confusion matrix helped us understand model predictions better.

Hence, The model shows good results and can be used as a tool for predicting loan approvals. Using This model can assist financial institutions in making data-driven, consistent, and faster loan decisions.