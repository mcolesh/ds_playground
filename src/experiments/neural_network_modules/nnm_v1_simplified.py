from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, Y = data.data, data.target

"""let's preprocess, normalize and create the model"""
from sklearn.model_selection import StandardScaler, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test.shape)
