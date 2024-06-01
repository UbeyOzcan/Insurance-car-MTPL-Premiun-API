import numpy as np
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices

df = pd.read_csv('../data/frenchmtpl_clean.csv', sep=';')

# Load modules and data
y, X = dmatrices('ClaimNb ~ VehPower + Area', data=df, return_type='dataframe')

print(y)
print(X)



