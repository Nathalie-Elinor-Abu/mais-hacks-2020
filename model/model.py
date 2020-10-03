import numpy as np
import pandas as pd
import os

data = pd.read_csv("../data/mbti.csv")

# printing the first 10 lines
print(data.head(10))
print(data.columns)
print(data.shape) # 8765 rows, 2 columns

# finding all the personality types
types = list(np.unique(data.type.values))
print("There are", types.shape[0], "personality types")
print(*types, sep='\n')
