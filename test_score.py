from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.5, random_state = 1)
print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)
logisticRegr = LogisticRegression(max_iter=2500)
logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test[0].reshape(1, -1))
logisticRegr.predict(X_test[0:10])
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)