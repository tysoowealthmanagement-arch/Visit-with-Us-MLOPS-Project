# Data Preparation
# Clean the raw data and split it into train/test sets.

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# drop columns we don't need for modeling
df = df.drop(columns=["CustomerID"])
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# fix a typo found in the Gender column during EDA
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print("Train shape:", Xtrain.shape)
print("Test shape:", Xtest.shape)
