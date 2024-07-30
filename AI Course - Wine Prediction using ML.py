import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv('WineQT.csv')

# Separate the data and Label
X = wine_dataset.drop(['quality','Id'], axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Model training using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy:', test_data_accuracy)

# F1 score on test data
test_data_f1_score = f1_score(Y_test, X_test_prediction)
print('F1 Score:', test_data_f1_score)

# Input data for prediction (with the missing feature added)
input_data = np.array([7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5])

# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data.reshape(1, -1)

# Making prediction using the trained model
prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
