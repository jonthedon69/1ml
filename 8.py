# PGM-8: Decision Tree Classifier on Breast Cancer Dataset

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the Breast Cancer dataset
data = pd.read_csv(r'C:\Users\Priyam\OneDrive\Desktop\AIML\Breast Cancer Dataset.csv')  # <-- Update path if necessary

# Display basic info
pd.set_option('display.max_columns', None)
print("\nFirst 5 rows of data:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nDataset Info:")
print(data.info())

# Unique values in diagnosis
print("\nUnique values in diagnosis:", data['diagnosis'].unique())

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Remove 'id' column and map diagnosis to 0 (Benign) and 1 (Malignant)
df = data.drop(['id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe().T)

# Preparing features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Splitting dataset into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting the Decision Tree
plt.figure(figsize=(20, 12))
plot_tree(model, 
          filled=True, 
          feature_names=X.columns, 
          class_names=['Benign', 'Malignant'],
          rounded=True, 
          fontsize=10)
plt.title('Decision Tree - Breast Cancer Diagnosis', fontsize=16)
plt.show()

# (Optional) Exporting the tree using Graphviz (only if Graphviz is installed)
try:
    from IPython.display import Image
    import pydotplus

    dot_data = export_graphviz(model, out_file=None,
                               feature_names=X.columns,
                               class_names=['Benign', 'Malignant'],
                               rounded=True, proportion=False, precision=2, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))
except:
    print("\nGraphviz not installed or not found. Skipping advanced tree visualization.")

# Calculate Information Gain for each feature (manually)
import math

def entropy(column):
    counts = column.value_counts()
    probabilities = counts / len(column)
    return -sum(probabilities * np.log2(probabilities))

def conditional_entropy(data, feature, target):
    feature_values = data[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return weighted_entropy

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    feature_entropy = conditional_entropy(data, feature, target)
    return total_entropy - feature_entropy

print("\nInformation Gain for each feature:")
for feature in X.columns:
    ig = information_gain(df, feature, 'diagnosis')
    print(f"{feature}: {ig:.4f}")

# Predict a new sample
new_sample = [[12.5, 19.2, 80.0, 500.0, 0.085, 0.1, 0.05, 0.02, 0.17, 0.06,
               0.4, 1.0, 2.5, 40.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.003,
               16.0, 25.0, 105.0, 900.0, 0.13, 0.25, 0.28, 0.12, 0.29, 0.08]]

new_pred = model.predict(new_sample)

# Output the prediction
if new_pred[0] == 0:
    print("\nPrediction for new sample: Benign")
else:
    print("\nPrediction for new sample: Malignant")
