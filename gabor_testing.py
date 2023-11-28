import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from PIL import Image
import gabor_filter
import otsu_thresholding
import other_funcs

input_img_path = input("Enter Path for Image: ")

# Background mask seperation
inputImage = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(inputImage, 20, 255, cv2.THRESH_BINARY)
se = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
erodedmask = cv2.erode(mask, se, iterations=1)
erodedmask = np.uint8(erodedmask)
img = Image.open(input_img_path)
img_array = np.array(img)

#Green channel seperation
green_channel = img_array[:, :, 1]

#Applying CLAHE based on hyperparameter tuning
clahe = cv2.createCLAHE(clipLimit = 8, tileGridSize=(8,8))
cl_img = clahe.apply(green_channel)

#Gamma correction
im_new , lose = other_funcs.replace_black_ring(cl_img,erodedmask)
im_new = 255-im_new

#Gabor Filtering 
wavelengths = [17]
number_of_orientations = 24
aspect_ratio = 0.5
bandwidth = 1

gabor_images = gabor_filter.apply_gabor(im_new, wavelengths, number_of_orientations, aspect_ratio, bandwidth)
thresholded_images = gabor_filter.apply_threshold(gabor_images)
gabor = thresholded_images[0]

#Morphological Operations + Otsu Thresholding
gabor = gabor.astype(np.uint8)
cell_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
Topen = cv2.morphologyEx(gabor,cv2.MORPH_OPEN,cell_disc)
Tclose = cv2.morphologyEx(Topen, cv2.MORPH_CLOSE, cell_disc)

tophat_img = (gabor - Tclose)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
bottomhat_img = cv2.erode(tophat_img, kernel)
bottomhat_img = cv2.dilate(bottomhat_img, kernel)
ret, thresh = cv2.threshold(bottomhat_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Hessian Matrix and EigenValues 
HessThin = hessian_matrix(thresh, sigma=1.2, order='rc',use_gaussian_derivatives=True)
EignThin = hessian_matrix_eigvals(HessThin) [1]
HessWide = hessian_matrix(thresh, sigma=4, order='rc', use_gaussian_derivatives=True)
EignWide = hessian_matrix_eigvals(HessWide) [1]

# Disable all runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Thresholding and Image Processing
val1 = otsu_thresholding.GlobalOtsu(1-EignWide)

thinN = cv2.normalize(1-EignThin,  None, 0, 255, cv2.NORM_MINMAX)
val1 = cv2.normalize(val1,  None, 0, 70, cv2.NORM_MINMAX)

test1 = otsu_thresholding.image_fusion(val1,thinN)
lOtsu = otsu_thresholding.LocalOtsu2(test1.astype(np.uint8))
final_img = otsu_thresholding.AreaThreshold(lOtsu,200)

#Binary conversion of the output
final_img[final_img!=0] = 255

#Saving the segmented image
image_pil = Image.fromarray(final_img)
save_path = input_img_path.split('.jpg')[0]+ '_SegmentedVessels.jpg'
image_pil.save(save_path)
print(f"Image Saved Successfully at {save_path}")

#Displaying the image for comparision
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_img, cmap='gray')
plt.title('Segmented Image (Gabor Filtered + Morphological)')
plt.axis('off')

plt.show()