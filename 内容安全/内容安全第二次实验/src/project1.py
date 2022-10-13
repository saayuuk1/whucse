import numpy as np
import cv2 as cv

#Load a image in grayscale
img = cv.imread('cat.jpeg', 0)
rows, cols= img.shape

#Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst1 = cv.warpAffine(img, M, (cols, rows))

#Scale
M = np.float32([[0.6, 0, 0], [0, 1.4, 0]])
dst4 = cv.warpAffine(img, M, (cols, rows))

#Rotation
#cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
dst2 = cv.warpAffine(img, M, (cols, rows))

#Shear in x direction
M = np.float32([[1, 0.4, 0], [0, 1, 0]])
dst5 = cv.warpAffine(img, M, (cols, rows))

#Shear in y direction
M = np.float32([[1, 0, 0], [0.4, 1, 0]])
dst6 = cv.warpAffine(img, M, (cols, rows))

#Reflect about origin
dst7 = cv.flip(img, -1)

#Reflect about x-axis
dst8 = cv.flip(img, 0)

#Reflect about y-axis
dst9 = cv.flip(img, 1)

#AffineTransform
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
dst3 = cv.warpAffine(img, M, (cols, rows))

cv.imshow('original', img)
cv.imshow('rotation', dst2)
cv.imshow('translation', dst1)
cv.imshow('Affine Transformation', dst3)
cv.imshow('Scale about origin', dst4)
cv.imshow('Shear in x direction', dst5)
cv.imshow('Shear in y direction', dst6)
cv.imshow('Reflect about origin', dst7)
cv.imshow('Reflect about x-axis', dst8)
cv.imshow('Reflect about y-axis', dst9)


cv.waitKey(0)