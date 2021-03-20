import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import array
from numpy import uint8
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

## predrawn digit 5
DRAWN_DIGIT = array([[ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  9, 12, 10, 11,  0,  0],
       [ 0,  0, 15,  0,  0,  0,  0,  0],
       [ 0,  0, 12, 12, 15,  2,  0,  0],
       [ 0,  0,  0,  0,  0, 15,  0,  0],
       [ 0,  0,  2,  0,  0,  9,  0,  0],
       [ 0,  0, 10, 12, 12, 10,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)

print(DRAWN_DIGIT)
print(DRAWN_DIGIT.shape)

PREDICT_DATA = DRAWN_DIGIT.reshape((1,-1))[0]

print("DIGITA ARRAY FLATTENED")
print(PREDICT_DATA)

predicted = classifier.predict([PREDICT_DATA])
print("PREDICTION")
print(predicted)

plt.subplot(1, 1, 1)
plt.imshow(DRAWN_DIGIT, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()