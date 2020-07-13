import cv2 as cv
import numpy as np
import pandas as pd

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):

    x, y = pos

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv.CascadeClassifier('Nose18x15.xml')

img = cv.imread('Before.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

glasses = cv.imread('glasses.png',-1)
mustache = cv.imread('mustache.png',-1)

# cv.imshow('Jamie',gray)

eye = eye_cascade.detectMultiScale(gray, 1.1, 5)
for x,y,w,h in eye:
	# cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
	glasses = cv.resize(glasses, (h,w))
	overlay_image_alpha(img, glasses[:,:,0:3], (x,y), glasses[:,:,3]/255.0)
	
nose = nose_cascade.detectMultiScale(gray, 1.1, 5)
for x,y,w,h in nose:
	# cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	mustache = cv.resize(mustache, (h+w,w))
	h = int(h/2)
	w = int(w/2)
	overlay_image_alpha(img, mustache[:,:,0:3], (x-w,y+h), mustache[:,:,3]/255.0)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Convert into csv
prediction = np.array(img)
prediction = prediction.reshape((-1,3))
print(prediction.shape)

df = pd.DataFrame(data = prediction, columns=['Channel 1', 'Channel 2', 'Channel 3'])
df.to_csv('result.csv',index=False)
