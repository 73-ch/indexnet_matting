import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/nami/Documents/tester/boxies/image_062.png')
# img = img[...,::-1]
matte = cv2.imread('/Users/nami/Documents/background_removal_test/indexnet_matting/examples/mattes/image_062.png')
h,w,_ = img.shape
bg = np.full_like(img,255) #white background

img = img.astype(float)
bg = bg.astype(float)

matte = matte.astype(float)/255.0
img = cv2.multiply(img, matte)
bg = cv2.multiply(bg, 1.0 - matte)
outImage = cv2.add(img, bg)
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(outImage/255)
cv2.imwrite('/Users/nami/Documents/background_removal_test/indexnet_matting/examples/images/63.png', outImage)
# cv2.imshow('image', outImage/255)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.show()