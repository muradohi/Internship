#Import necessary libraries
import numpy as np
import pandas as pd

dataset= pd.read_csv('dataset/balanced_cirrhosis_dataset.csv')

#print(dataset.head())

# Display basic information about the dataset
print(dataset.info())

# Check for missing values
print(dataset.isnull().sum())
print(dataset.columns)
print(dataset.drop("Stage",axis=1))



