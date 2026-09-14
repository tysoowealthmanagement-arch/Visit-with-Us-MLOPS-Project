# Data Registration
# The dataset lives in this GitHub repo, so "registering" it just means
# checking that the file is there and looks correct before we use it.

import pandas as pd

df = pd.read_csv("tourism_project/data/tourism.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nTarget value counts:")
print(df["ProdTaken"].value_counts())
