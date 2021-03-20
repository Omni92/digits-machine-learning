import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

matplotlib.use('TkAgg')

img_path = './digit.jpeg'
img = cv2.imread(img_path, 0) # read image as grayscale.
img_reverted = cv2.bitwise_not(img)

new_img = img_reverted / 16

print(repr(new_img))

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
DRAWN_DIGIT = new_img

# DRAWN_DIGIT = array([[ 0,  0,  0,  0,  0,  0,  0,  0],
#        [ 0,  0,  9, 12, 10, 11,  0,  0],
#        [ 0,  0, 15,  0,  0,  0,  0,  0],
#        [ 0,  0, 12, 12, 15,  2,  0,  0],
#        [ 0,  0,  0,  0,  0, 15,  0,  0],
#        [ 0,  0,  2,  0,  0,  9,  0,  0],
#        [ 0,  0, 10, 12, 12, 10,  0,  0],
#        [ 0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)

# DRAWN_DIGIT = array([[ 0,  0,  1,  5,  3,  0,  0,  0],
#        [ 0,  4, 11,  5, 11,  1,  0,  0],
#        [ 0,  7,  1,  0,  7,  2,  0,  0],
#        [ 0,  0,  0,  0,  8,  3,  0,  0],
#        [ 0,  0,  0,  6,  9,  1,  0,  0],
#        [ 0,  5,  8,  5,  0,  0,  0,  0],
#        [ 2, 15,  9,  5,  5,  5,  5,  5],
#        [ 1,  5,  4,  4,  4,  4,  4,  4]], dtype=uint8)

print(DRAWN_DIGIT)
print(DRAWN_DIGIT.shape)

PREDICT_DATA = DRAWN_DIGIT.reshape((1,-1))[0]

print(PREDICT_DATA)

predicted = classifier.predict([PREDICT_DATA])
print("PREDICTION")
print(predicted)

plt.subplot(1, 1, 1)
plt.imshow(DRAWN_DIGIT, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()