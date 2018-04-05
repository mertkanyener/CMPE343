import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))


df_wineBest = df_wine.loc[:, ('Proline','Flavanoids')].values
df_wineWorst= df_wine.loc[:, ('Nonflavanoid phenols', 'Ash')].values


XB= df_wineBest
XW= df_wineWorst
stdsc = StandardScaler()
X_stdB = stdsc.fit_transform(XB)
X_stdW = stdsc.fit_transform(XW)


XB_train, XB_test, y_train, y_test = train_test_split(X_stdB, y, test_size=0.3,
                                                      random_state=0,
                                                      stratify=y)

XW_train, XW_test, y_train, y_test = train_test_split(X_stdW, y, test_size=0.3,
                                                      random_state=0,
                                                      stratify=y)

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(XB_train, y_train)
plot_decision_regions(X_stdB, y, classifier=lr, test_idx=range(105, 150))
plt.xlabel('Proline')
plt.ylabel('Flavanoids')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/BestLR')
plt.clf()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(XW_train, y_train)
plot_decision_regions(X_stdW, y, classifier=lr, test_idx=range(105, 150))
plt.xlabel('Nonflavanoid phenols')
plt.ylabel('Ash')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plots/WorstLR')

"""
svm_model = SVC(kernel='linear', C=1.0, random_state=1)
svm_model.fit(XB_train, y_train)
plot_decision_regions(X_stdB, y, classifier=svm_model, test_idx=range(105, 150))
plt.xlabel('Proline')
plt.ylabel('Flavanoids')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""