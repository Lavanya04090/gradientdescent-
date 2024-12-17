import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\Users\KIIT\Desktop\bot docs\rawdatagradient.csv"
data = pd.read_csv(file_path)

label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])

X = data[['Age', 'City']].values
y = data['Income'].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

X = np.c_[np.ones(X.shape[0]), X]

theta = np.zeros(X.shape[1])

learning_rate = 0.01
iterations = 1000
m = len(y)

for _ in range(iterations):
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    theta -= learning_rate * gradient

print(f"Learned parameters (theta): {theta}")