import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.impute import SimpleImputer

# Loading Data from Churn_Modelling.csv in pandas DataFrame
cm = pd.read_csv("Churn_Modelling.csv", encoding="ISO-8859-1")

# Appointing Features and Target
X = cm[['RowNumber', 'Surname', 'CustomerId', 'Geography', 'Gender', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
Y = cm['Exited']

# Splitting the data into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Create TF-IDF vectorizers for 'Surname', 'Geography', and 'Gender'
surname_vectorizer = TfidfVectorizer()
geography_vectorizer = TfidfVectorizer()
gender_vectorizer = TfidfVectorizer()

# Fit and transform the 'Surname' feature
X_train_surname_tfidf = surname_vectorizer.fit_transform(X_train['Surname'])
X_test_surname_tfidf = surname_vectorizer.transform(X_test['Surname'])

# Fit and transform the 'Geography' feature
X_train_geography_tfidf = geography_vectorizer.fit_transform(X_train['Geography'])
X_test_geography_tfidf = geography_vectorizer.transform(X_test['Geography'])

# Fit and transform the 'Gender' feature
X_train_gender_tfidf = gender_vectorizer.fit_transform(X_train['Gender'])
X_test_gender_tfidf = gender_vectorizer.transform(X_test['Gender'])

# Concatenate TF-IDF features with the original features
X_train_final = hstack((X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values,
                       X_train_surname_tfidf, X_train_geography_tfidf, X_train_gender_tfidf), format='csr')

X_test_final = hstack((X_test[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values,
                      X_test_surname_tfidf, X_test_geography_tfidf, X_test_gender_tfidf), format='csr')

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform on the training data as well as testing data
X_train_final = imputer.fit_transform(X_train_final)
X_test_final = imputer.transform(X_test_final)

# Training the model
model = RandomForestClassifier()

# Training the RandomForestClassifier Model with the training data
model.fit(X_train_final, Y_train)

# Prediction on training data
Prediction_on_Training_data = model.predict(X_train_final)
accuracy_on_Training_data = accuracy_score(Y_train, Prediction_on_Training_data)

print("The accuracy score on Test Data is : " + str(accuracy_on_Training_data * 100) + "%") 

# Prediction on test data
Prediction_on_Test_data = model.predict(X_test_final)
accuracy_on_Test_data = accuracy_score(Y_test, Prediction_on_Test_data)

print("The accuracy score on Test Data is : " + str(accuracy_on_Test_data * 100) + "%") 

# Function to preprocess and predict on custom input
def predict_custom_input(custom_input):
    # Preprocess custom input
    custom_df = pd.DataFrame(custom_input, columns=X.columns)
    
    # Check if 'Surname' is in the input, and provide a default value if not
    if 'Surname' not in custom_df.columns:
        custom_df['Surname'] = 'DefaultSurname'
    
    # Transform the 'Surname' feature
    custom_surname_tfidf = surname_vectorizer.transform(custom_df['Surname'].astype(str))  

    # Transform the 'Geography' feature
    custom_geography_tfidf = geography_vectorizer.transform(custom_df['Geography'])
    
    # Transform the 'Gender' feature
    custom_gender_tfidf = gender_vectorizer.transform(custom_df['Gender'])
    
    # Concatenate TF-IDF features with the original features
    custom_final = hstack((custom_df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values,
                           custom_surname_tfidf, custom_geography_tfidf, custom_gender_tfidf), format='csr')
    
    # Impute missing values
    custom_final = imputer.transform(custom_final)
    
    # Make predictions
    predictions = model.predict(custom_final)
    
    return predictions

custom_input = {
    'CreditScore': 750,
    'Age': 40,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 60000,
    'Surname':'Shukla',
    'Geography': 'France',
    'Gender': 'Male'
}

predictions = predict_custom_input([custom_input])

if predictions[0] == 0:
    print("On the Custom Input Given the model predicts that the customer will 'not exit'.")
else:
    print("On the Custom Input Given the model predicts that the customer will 'exit'.")
