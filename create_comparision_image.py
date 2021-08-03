import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('/ver2/softening=10^-2/density_histgram/tEnd=4.png')
img2 = cv2.imread('/ver2/softening=10^-3/density_histgram/tEnd=4.png')

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# blended = cv2.addWeighted(src1=img1,alpha=0.7,src2=img2,beta=0.3,gamma=0)
# plt.imshow(blended)