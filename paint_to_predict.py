from tkinter import *
from PIL import Image, ImageDraw
import cv2
from sklearn import datasets, svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

digits = datasets.load_digits()

logisticRegr = LogisticRegression(max_iter=2500)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
logisticRegr.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

class Paint(object):
    LINE_WIDTH = 75

    def __init__(self):
        self.old_x = None
        self.old_y = None

        self.root = Tk()

        self.prediction_button = Button(self.root, text='Predict', command=self.predict)
        self.prediction_button.grid(row=0, column=0)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.create_image()
        self.root.mainloop()

    def create_image(self):
        self.image = Image.new("RGB", (600, 600), (255, 255, 255))
        self.image_draw = ImageDraw.Draw(self.image)

    def predict(self):
        resized_img = self.image.resize((8, 8))
        resized_img.save("test.jpeg")

        img = cv2.imread("./test.jpeg", cv2.IMREAD_GRAYSCALE)  # read image as grayscale.
        img_reverted = cv2.bitwise_not(img)

        new_img = img_reverted / 16

        data = new_img.reshape((1, -1))[0]

        c = 0
        for i in data:
            if i < 3:
                data[c] = 0
            c += 1

        print(data)

        predicted = logisticRegr.predict([data])

        print(predicted)

        self.c.delete("all")
        self.create_image()

        plt.subplot(1, 1, 1)
        plt.imshow(new_img, cmap=plt.cm.gray_r, interpolation='nearest')

        plt.show()

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width= self.LINE_WIDTH, fill="black",
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.image_draw.line([self.old_x, self.old_y, event.x, event.y], (0, 0, 0), self.LINE_WIDTH)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
