import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart_disease_data.csv")

# One-Hot Encoding for Categorial Variables 
cp_dummies = pd.get_dummies(data['cp'], prefix = "cp")
thal_dummies = pd.get_dummies(data['thal'], prefix = "thal")
slope_dummies = pd.get_dummies(data['slope'], prefix = "slope")
data_all = pd.concat([data, cp_dummies, thal_dummies, slope_dummies], axis = 1)
data_all = data_all.drop(columns = ['cp', 'thal', 'slope'])

# Feature Scaling: Min/Max Normalization
y = data_all.target.values
x_data = data_all.drop(['target'], axis = 1)
X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# Train-test split: assess model using test partition, data the model hasn't seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)