from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Load the training dataset
train_dataset = pd.read_csv('adult-training.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
X_train = train_dataset.drop('income', axis=1)
y_train = train_dataset['income']

# Load the testing dataset
test_dataset = pd.read_csv('adult-test.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
X_test = test_dataset.drop('income', axis=1)
y_test = test_dataset['income']

# Drop rows with missing labels in y_test
y_test = y_test.dropna()

# Clean labels in y_train and y_test
y_train = y_train.str.replace(r'\.', '')  # Remove periods from labels
y_test = y_test.str.replace(r'\.', '')    # Remove periods from labels

# Apply label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Convert numeric columns to strings
numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
X_train[numeric_columns] = X_train[numeric_columns].astype(str)
X_test[numeric_columns] = X_test[numeric_columns].astype(str)

# Concatenate the datasets
concatenated = pd.concat([X_train, X_test], axis=0)

# Perform one-hot encoding
encoder = OneHotEncoder()
encoded = encoder.fit_transform(concatenated)

# Split the encoded data back into training and testing sets
X_train_encoded = encoded[:len(X_train)]
X_test_encoded = encoded[len(X_train):]

# Create and train the random forest model
clf = RandomForestClassifier()
clf.fit(X_train_encoded, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_encoded)

# Verify the shapes of y_test and y_pred
print("X_test shape:", X_test.shape)
print("y_pred shape:", y_pred.shape)
if y_test.shape == y_pred.shape:
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
else:
    print("Error: Number of samples in y_test and y_pred is inconsistent.")
