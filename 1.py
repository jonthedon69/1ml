# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'C:\Users\Priyam\OneDrive\Desktop\AIML\housing.csv') 

# Display first few rows
print(df.head())

# Check shape of dataset
print("Dataset Shape:", df.shape)

# Basic info
df.info()

# Number of unique values in each column
print(df.nunique())

# Check missing values
print(df.isnull().sum())

# Check duplicated records
print("Duplicated Rows:", df.duplicated().sum())

# Handling missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Feature Engineering: Convert some features to int
for col in ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households']:
    df[col] = df[col].astype('int')

# Descriptive statistics
print(df.describe().T)

# Select numerical columns
Numerical = df.select_dtypes(include=[np.number]).columns
print("Numerical Columns:", Numerical)

# Uni-variate Analysis: Histograms
for col in Numerical:
    plt.figure(figsize=(10, 6))
    df[col].plot(kind='hist', title=col, bins=60, edgecolor='black')
    plt.ylabel('Frequency')
    plt.show()

# Uni-variate Analysis: Box plots
for col in Numerical:
    plt.figure(figsize=(6, 6))
    sns.boxplot(x=df[col], color='blue')
    plt.title(col)
    plt.ylabel(col)
    plt.show()
