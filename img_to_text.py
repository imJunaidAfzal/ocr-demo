import pytesseract
import cv2
img = cv2.imread('text3.png')

img = cv2.resize(img, (600, 360))
result = pytesseract.image_to_string(img)
print(result)
cv2.imshow('Result', img)
cv2.waitKey(0)
print('end')
