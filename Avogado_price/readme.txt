Outlier detection and removal
We have a significant problems with outliers in both data sets:

most of the distributions are not normal;

huge outliers;

higly right-skeved data in Avocado Prices data set;

a lot of outliers.

Tukey’s (1977) technique is used to detect outliers in skewed or non bell-shaped data since it makes no distributional assumptions. However, Tukey’s method may not be appropriate for a small sample size. The general rule is that anything not in the range of (Q1 - 1.5 IQR) and (Q3 + 1.5 IQR) is an outlier, and can be removed.

Inter Quartile Range (IQR) is one of the most extensively used procedure for outlier detection and removal.

Procedure:

Find the first quartile, Q1.
Find the third quartile, Q3.
Calculate the IQR. IQR = Q3-Q1.
Define the normal data range with lower limit as Q1–1.5 IQR and upper limit as Q3+1.5 IQR.

/** Must Read data**/
https://www.kaggle.com/code/marcinrutecki/outlier-detection-methods

df5.loc[:, 'Total Volume':'WestTexNewMexico']:

.loc is used for label-based indexing.

The : before the comma means “select all rows”.

his is commonly done to extract features for machine learning — isolating numerical or relevant input variables (features) into X, which can then be passed to a model.