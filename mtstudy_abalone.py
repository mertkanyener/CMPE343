import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('abalone', header=None)
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
              'Viscera weight', 'Shell weight', 'Rings']

df = df.groupby('Rings').filter(lambda x: len(x) > 1)


"""
# Mapping ordinal features
sex_mapping = {'M': 1,
               'F': 2,
               'I': 3}
df['Sex'] = df['Sex'].map(sex_mapping)
"""

X, y = df.iloc[:, :8].values, df.iloc[:, 8].values
sex_le = LabelEncoder()
X[:, 0] = sex_le.fit_transform(X[:, 0])


#y = np.array(y, dtype='float')
#y += 1.5



# Performing one-hot encoding on nominal features

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()[:, 1:]  # multicollinearity guard for onehotencoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)


# Assesssing feature importance with Random Forests

forest = RandomForestClassifier(n_estimators=128,
                                random_state=1)
df_columns = ['Sex_male', 'Sex_infant', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
              'Viscera weight', 'Shell weight']
forest.fit(X_train, y_train)
feat_labels = df_columns
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances: ")
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

df_best = df.loc[:, ('Shell weight', 'Shucked weight')]


XB_train, XB_test, y_train, y_test = train_test_split(df_best, y)




