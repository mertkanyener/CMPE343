import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'


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


def log_reg(X_train_std, y_train,  X_combined_std, y_combined, i, j):
    lr = LogisticRegression(C=100.0, random_state=1)
    lr.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(100, 150))
    plt.xlabel(str(i))
    plt.ylabel(str(j))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/' + str(i) + 'v' + str(j) + '_logreg', dpi=300)
    print('plot', i, 'v', j, '_logreg saved')
    plt.clf()


def rand_forest(X_train, y_train, X_combined, y_combined, i, j):
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined,
                                classifier=forest, test_idx=range(105, 150))
    plt.xlabel(str(i))
    plt.ylabel(str(j))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/' + str(i) + 'v' + str(j) + '_rforest')
    print('plot', i, 'v', j, '_rforestsaved')
    plt.clf()


def svm(X_train_std, y_train, X_combined_std, y_combined, i, j):
    svm_model = SVC(kernel='linear', C=1.0, random_state=1)
    svm_model.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                                classifier=svm_model,
                                test_idx=range(105, 150))
    plt.xlabel(str(i))
    plt.ylabel(str(j))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/' + str(i) + 'v' + str(j) + '_svm', dpi=300)
    print('plot', i, 'v', j, '_svm saved')
    plt.clf()


def main_loop(i=6, j=7):

    df_wine = pd.read_csv(url,
                          header=None)
    while i+1 < 13:
        if j <= 13:
            X = df_wine.loc[:, (i, j)].values
            y = df_wine.loc[:, 0].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=1, stratify=y)

            sc = StandardScaler()
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            X_combined_std = np.vstack((X_train_std, X_test_std))
            y_combined = np.hstack((y_train, y_test))
            X_combined = np.vstack((X_train, X_test))

            svm(X_train_std, y_train, X_combined_std, y_combined, i, j)
            rand_forest(X_train, y_train, X_combined, y_combined, i, j)
            log_reg(X_train_std, y_train, X_combined_std, y_combined, i, j)
            j += 1

        else:
            i += 1
            j = i + 1




#main_loop()

df_wine = pd.read_csv(url, header=None)

