import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer

# Load the dataset (replace 'graduate_admissions.csv' with the actual filename)
admissions_data = pd.read_csv('admission_data.csv')

# Data preprocessing
admissions_data.columns = admissions_data.columns.str.strip()  # Handle spaces in column names

# Rename the target column to remove spaces
admissions_data = admissions_data.rename(columns={'Chance of Admit ': 'Chance of Admit'})

# Convert 'Chance of Admit' into discrete bins (customize the number of bins)
k_bins = 5  # Adjust the number of bins as needed
admissions_data['Admit_Category'] = pd.cut(admissions_data['Chance of Admit'], bins=k_bins, labels=False)

# Label encoding for the 'Research' column
le = LabelEncoder()
admissions_data['Research'] = le.fit_transform(admissions_data['Research'])

# Selecting relevant columns for features and target
X = admissions_data.drop(['Chance of Admit', 'Admit_Category'], axis=1)
y = admissions_data['Admit_Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on the test set: {accuracy}")

# Now you can use the trained model (gnb) to make predictions for new data.
# For example:
new_data = np.array([[320, 110, 3, 4, 4.5, 8.5, 1]])  # Replace with your own values
new_pred = gnb.predict(new_data)
print(f"Predicted admission category for new data: {new_pred[0]}")
