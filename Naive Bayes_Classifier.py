import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\locha\OneDrive\Desktop\NLP\mushrooms.csv")
print(df.shape)
print(df.head())

le = LabelEncoder()
df_encoded = df.apply(le.fit_transform, axis=0)
print(df_encoded.head())

df = df_encoded.values
x = df[:, 1:]
y = df[:, 0]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

def prior_probability(y, class_value):
    m = y.shape[0]
    pp = np.sum(y == class_value) / m
    return pp

def conditional_prob(x, y, feature_col, feature_value, class_value):
    x_filtered = x[y == class_value]
    res = np.sum(x_filtered[:, feature_col] == feature_value)
    return res / x_filtered.shape[0]

def likelihood(x, y, row, class_value):
    prob = 1.0
    for i in range(x.shape[1]):
        prob *= conditional_prob(x, y, i, row[i], class_value)
    return prob

def predict(x, y, x_row):
    probs = []
    classes = np.unique(y)
    for c in classes:
        prior = prior_probability(y, c)
        like = likelihood(x, y, x_row, c)
        probs.append(prior * like)
    return classes[np.argmax(probs)]

y_pred = []
for row in x_test:
    y_pred.append(predict(x_train, y_train, row))

y_pred = np.array(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)