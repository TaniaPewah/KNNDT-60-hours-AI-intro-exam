import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
XY9 = pd.read_csv('./Dataset 9.csv', header=None)
X = XY9.iloc[:, 1:-1].to_numpy()
Y = XY9.iloc[:, -1].to_numpy()

sss = StratifiedShuffleSplit(n_splits=1, train_size=200, test_size=100)
train_rows, test_rows = sss.split(X, Y).__next__()
train_rows = sorted(train_rows)
test_rows = sorted(test_rows)
X_train, Y_train = [X[j] for j in train_rows], [Y[j] for j in train_rows]
X_test, Y_test = [X[j] for j in test_rows], [Y[j] for j in test_rows]


XY_train = pd.DataFrame(X_train)
XY_train[len(XY_train.columns)] = pd.Series(Y_train)
XY_train.to_csv('train_9.csv')
XY_test = pd.DataFrame(X_test)
XY_test[len(XY_test.columns)] = pd.Series(Y_test)
XY_test.to_csv('test_9.csv')


XY12 = pd.read_csv('./Dataset 12.csv', header=None)
X = XY12.iloc[:, 1:-1].to_numpy()
Y = XY12.iloc[:, -1].to_numpy()

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, train_size=200, test_size=100)
train_rows, test_rows = sss.split(X, Y).__next__()
train_rows = sorted(train_rows)
test_rows = sorted(test_rows)
X_train, Y_train = [X[j] for j in train_rows], [Y[j] for j in train_rows]
X_test, Y_test = [X[j] for j in test_rows], [Y[j] for j in test_rows]

XY_train = pd.DataFrame(X_train)
XY_train[len(XY_train.columns)] = pd.Series(Y_train)
XY_train.to_csv('train_12.csv')
XY_test = pd.DataFrame(X_test)
XY_test[len(XY_test.columns)] = pd.Series(Y_test)
XY_test.to_csv('test_12.csv')

