# AdmissionPredictionSystem

This project employs the Gaussian Naive Bayes algorithm to predict admission categories based on a dataset. The dataset includes features such as GRE Score, TOEFL Score, University Rating, SOP Score, LOR Score, CGPA, and Research, with the target variable being the admission category.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Dataset

The dataset is loaded from the 'admission_data.csv' file. Ensure that the actual dataset filename is used if different.

## Data Preprocessing

- Columns with spaces in their names are handled.
- The target column 'Chance of Admit' is converted into discrete bins using `pd.cut`.
- Label encoding is applied to the 'Research' column using `LabelEncoder`.

## Model Training

The dataset is split into training and testing sets using the `train_test_split` method. A Gaussian Naive Bayes model is then trained on the training set.

```python
# Code snippet for model training
# ...

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on the test set: {accuracy}")
