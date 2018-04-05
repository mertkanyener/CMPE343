import pandas as pd


url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)


X = df_wine.loc[:, (1,3)].values
print(X)
#print(df_wine.head(20))

