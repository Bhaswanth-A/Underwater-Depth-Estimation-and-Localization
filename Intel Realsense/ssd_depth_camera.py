import numpy
import cv2

from matplotlib import pyplot as plt
from matplotlib import cm

left = cv2.imread(
    "/home/bhaswanth/Pictures/depth-cam/bottle_l_2_Infrared.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(
    "/home/bhaswanth/Pictures/depth-cam/bottle_r_2_Infrared.png", cv2.IMREAD_GRAYSCALE)

h,  w = left.shape[:2]

width = int(w * 0.3)  # 0.3
height = int(h * 0.3)  # 0.3

left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
# img1 = cv2.GaussianBlur(img1,(5,5),0)
right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
# img2 = cv2.GaussianBlur(img2,(5,5),0)


# left = cv2.imread(
#     "/home/bhaswanth/Downloads/source/l_active.png", cv2.IMREAD_GRAYSCALE)
# right = cv2.imread(
#     "/home/bhaswanth/Downloads/source/r_active.png", cv2.IMREAD_GRAYSCALE)

fx = 942.8        # lense focal length
baseline = 54.8   # distance in mm between the two cameras
disparities = 160  # num of disparities to consider
block = 3       # block size to match
units = 0.512     # depth units, adjusted for the output to fit in one byte

# window_size = 3

sbm = cv2.StereoSGBM_create(numDisparities=disparities,
                            blockSize=block)

# sbm = cv2.StereoSGBM_create(
#     minDisparity=-1,
#     numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
#     blockSize=window_size,
#     P1=8 * 3 * window_size,
#     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#     P2=32 * 3 * window_size,
#     disp12MaxDiff=12,
#     uniquenessRatio=10,
#     speckleWindowSize=50,
#     speckleRange=32,
#     preFilterCap=63,
#     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
# )

# calculate disparities
disparity = sbm.compute(left, right)
valid_pixels = disparity > 0

# calculate depth data
depth = numpy.zeros(shape=left.shape).astype("uint8")
depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

# visualize depth data
depth = cv2.equalizeHist(depth)
colorized_depth = numpy.zeros((left.shape[0], left.shape[1], 3), dtype="uint8")
temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
colorized_depth[valid_pixels] = temp[valid_pixels]
plt.imshow(colorized_depth)
plt.show()
