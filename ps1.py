import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
data.drop(columns="Unnamed: 32", inplace=True)

data.replace(['M', 'B'], [1, 0], inplace=True)
ratio = 424
X = data.drop(columns="diagnosis", axis=1)
Y = data['diagnosis']
X = X.values
Y = Y.values
X_train = X[:ratio]
Y_train = Y[:ratio]
X_test = X[ratio:]
Y_test = Y[ratio:]


scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)


def accuracy(y, y_pred):
    return sum(1 for a, b in zip(y, y_pred) if a == b)/len(y)


class LogisticR:
    def __init__(self, lr=0.01, it=10000):
        self.lr = lr
        self.it = it

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def init_wt(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.init_wt(n_features)

        for i in range(self.it):
            linear_combination = np.dot(X, self.weights) + self.bias

            predictions = self.sigmoid(linear_combination)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:

                # cost = (-1/n_samples) * np.sum(y*np.log(predictions)+(1-y)*np.log(1-predictions))
                # print(f"{'*'*int(i/self.it*100)}", end='')
                print()

    def predict(self, X):
        linear_combination = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_combination)
        return [1 if p >= 0.5 else 0 for p in predictions]


custom_model = LogisticR(lr=0.0001, it=100000)


custom_model.fit(X_train, Y_train)


y_pred = custom_model.predict(X_test)


accuracy_custom = accuracy(Y_test, y_pred)
print("Accuracy:", accuracy_custom * 100, "%")
