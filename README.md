# Predicting-Customer-Churn-with-Machine-Learning
In today's highly competitive business landscape, retaining customers is often as important as acquiring new ones. Customer churn, the phenomenon where customers discontinue their association with a business, can have significant financial implications.

# Libraries used in this project

Using pandas from sklearn Library

Using train_test_split from sklearn Library

Using RandomForestClassifier from sklearn Library as a ML model in this project

Using accuracy_score from sklearn.metrics Library

Using TfidfVectorizer from sklearn.feature_extraction.text Library 

Using hstack from scipy.sparse Library

Using SimpleImputer from sklearn.impute

# Data Used in this project is from kraggle.com

# Step-1 : 
Loading Data from Churn_Modelling.csv in pandas DataFrame , Since Data is in not in the form of csv UTF-8 Format so we convert it into ISO-8859-1 format

# Step-2 : 
Replacing null values with null string as data have some null values

# Step-3 : 
Appointing Features and Target for the dataset as datapoints

# Step-4 : 
Splitting the data into test and train sets

# Step-5 : 
Convert Y_train and Y_test values as integers as datatype of the Y value is in object form as declared in shape

# Step-6 : 
Training the RandomForestClassifier Model with the training data

# Step-7 : 
Prediction on Training data and Test Data

# Step-8 :
Making Prediction Using Custom Input
