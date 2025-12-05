import cv2 as cv
import numpy as np

img=cv.imread("opencv/pic4.png")
cv.imshow("org",img)


#CONVERSION TO GRAY SCALE
GRay=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",GRay)



#COUNTING THE NO OF RED PIXELS
HSV=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("hsv",HSV)

lower_red1=np.array([0,120,70])
upper_red1=np.array ([10,255,255])

lower_red2=np.array([170,120,70])
upper_red2=np.array([180,255,255])

mask1=cv.inRange(HSV,lower_red1,upper_red1)
mask2=cv.inRange(HSV,lower_red2,upper_red2)

masks=mask1+mask2

red_pixels=cv.countNonZero(masks)
print(f"The no of Red Pixels are {red_pixels}")

cv.imshow("masks",masks)




import cv2 as cv
import numpy as np


img = cv.imread("opencv/pic4.png")  
if img is None:
    raise FileNotFoundError("Image not found, check the path bro.")

# (optional) resize if image is huge
# img = cv.resize(img, (800, 600))


# img.shape -> (height, width, 3)
pixels = img.reshape((-1, 3))         # (num_pixels, 3)
pixels = np.float32(pixels)           # kmeans needs float32


# criteria = (type, max_iter, epsilon)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            10,        # stop after 10 iterations max
            1.0)       # or when centers move less than 1.0

K = 1  # number of clusters (1 dominant color)
attempts = 10  # run kmeans 10 times with different random starts


compactness, labels, centers = cv.kmeans(
    pixels,                  # data: all pixels
    K,                       # K = number of color groups
    None,                    # no initial labels
    criteria,                # stopping criteria
    attempts,                # how many times to try
    cv.KMEANS_RANDOM_CENTERS # how to pick initial centers
)

# centers are the cluster colors in float32, convert back to 0–255
centers = np.uint8(centers)       # shape: (K, 3)


# labels is size (num_pixels, 1) with values 0..K-1
counts = np.bincount(labels.flatten())  # how many pixels in each cluster
dominant_idx = np.argmax(counts)        # index of biggest cluster
dominant_color = centers[dominant_idx]  # [B, G, R]

B, G, R = dominant_color
print("Dominant color (BGR):", dominant_color)
print("Dominant color (RGB):", (int(R), int(G), int(B)))


swatch = np.zeros((200, 200, 3), dtype='uint8')
swatch[:] = dominant_color  # fill with that color

cv.imshow("Original Image", img)
cv.imshow("Dominant Color", swatch)




cv.waitKey(0)






