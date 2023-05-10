import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the training dataset
train_dataset = pd.read_csv('adult-training.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
X_train = train_dataset.drop('income', axis=1)
y_train = train_dataset['income']

# Load the testing dataset
test_dataset = pd.read_csv('adult-test.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
X_test = test_dataset.drop('income', axis=1)
y_test = test_dataset['income']

# Drop rows with missing labels in y_test
missing_label_rows = y_test.isna()
X_test = X_test.loc[~missing_label_rows]
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

# Encode the categorical features using label encoding
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for feature in categorical_features:
    label_encoder = LabelEncoder()
    X_train_encoded[feature] = label_encoder.fit_transform(X_train[feature])
    X_test_encoded[feature] = label_encoder.transform(X_test[feature])

# Train the model
clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0)
clf.fit(X_train_encoded, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform hypothesis test
data = pd.DataFrame({'sex': X_test['sex'], 'income': y_test})
data = data.dropna()

contingency_table = pd.crosstab(data['sex'], data['income'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Interpret the results
alpha = 0.05
if p < alpha:
    print("There is evidence of an association between gender and income.")
else:
    print("There is no evidence of an association between gender and income.")