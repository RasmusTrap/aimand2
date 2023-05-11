import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
test_dataset = pd.read_csv('adult-test.csv', delimiter=',', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
X_test = test_dataset.drop('income', axis=1)
y_test = test_dataset['income']

# Create a contingency table
contingency_table = pd.crosstab(test_dataset['sex'], test_dataset['income'])

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the test statistics and p-value
print("Chi-square:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)