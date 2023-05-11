from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load the training dataset
train_dataset = pd.read_csv('adult-training.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Filter out rows with invalid data
train_dataset = train_dataset[train_dataset['age'] != '|1x3 Cross validator']

X_train = train_dataset.drop('income', axis=1)
y_train = train_dataset['income']

# Load the testing dataset
test_dataset = pd.read_csv('adult-test.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Filter out rows with invalid data
test_dataset = test_dataset[test_dataset['age'] != '|1x3 Cross validator']

X_test = test_dataset.drop('income', axis=1)
y_test = test_dataset['income']

# Drop rows with missing labels in y_test
y_test = y_test.dropna()

# Clean labels in y_train and y_test
y_train = y_train.str.replace(r'\.', '')  # Remove periods from labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load the training dataset
train_dataset = pd.read_csv('adult-training.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Filter out rows with invalid data
train_dataset = train_dataset[train_dataset['age'] != '|1x3 Cross validator']

X_train = train_dataset.drop('income', axis=1)
y_train = train_dataset['income']

# Load the testing dataset
test_dataset = pd.read_csv('adult-test.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Filter out rows with invalid data
test_dataset = test_dataset[test_dataset['age'] != '|1x3 Cross validator']

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

# Convert categorical columns to strings
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
X_train[categorical_columns] = X_train[categorical_columns].astype(str)
X_test[categorical_columns] = X_test[categorical_columns].astype(str)

# Apply one-hot encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns)

# Align the encoded training and testing datasets
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': ['auto']
}

# Create and train the random forest model
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train_encoded, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)