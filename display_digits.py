import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state = 1)

logisticRegr = LogisticRegression(max_iter=2500)
logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test[0].reshape(1, -1))
logisticRegr.predict(X_test[0:10])
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)

plt.subplot(321)
plt.imshow(digits.images[1791], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.subplot(322)
plt.imshow(digits.images[1792], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.subplot(323)
plt.imshow(digits.images[1793], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.subplot(324)
plt.imshow(digits.images[1794], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.subplot(325)
plt.imshow(digits.images[1795], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.subplot(326)
plt.imshow(digits.images[1796], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()