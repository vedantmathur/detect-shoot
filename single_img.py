import cv2
import matplotlib.pyplot as plt

imagePath = './data_input/image.jpg'
img = cv2.imread(imagePath)

print(img.shape)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_img, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

plt.show()