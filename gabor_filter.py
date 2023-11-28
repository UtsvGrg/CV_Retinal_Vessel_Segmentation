import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_images(images, title):
    num_images = len(images)
    plt.figure(figsize=(12, 6))
    for idx, lol in enumerate(images):
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(lol, cmap='gray')
        plt.title(f'{title} Image {idx + 1}')
        plt.axis('off')
    plt.show()



def gabor_kernel(wavelength, orientation, sigma, aspect_ratio):
    n_sigma = 2.5 * sigma / aspect_ratio
    n = int(np.floor(n_sigma))  # Size of the Gabor filter kernel

    x, y = np.meshgrid(range(-n, n + 1), range(-n, n + 1))
    y = -y  # Change the direction of y

    f = 2 * np.pi / wavelength
    b = 1 / (2 * sigma**2)
    a = b / np.pi

    xp = x * np.cos(orientation) + y * np.sin(orientation)
    yp = -x * np.sin(orientation) + y * np.cos(orientation)

    cos_func = np.cos((f * xp) - np.pi)
    kernel = a * np.exp(-b * (xp**2 + (aspect_ratio**2 * yp**2))) * cos_func

    # Normalize the kernel
    pos = np.sum(kernel[kernel > 0])
    neg = np.sum(np.abs(kernel[kernel < 0]))

    kernel[kernel > 0] /= pos
    kernel[kernel < 0] /= neg

    return kernel

def apply_gabor(image, wavelengths, number_of_orientations, aspect_ratio, bandwidth):
    slratio = (1 / np.pi) * np.sqrt(np.log(2) / 2.0 * ((2 ** bandwidth + 1) / (2 ** bandwidth - 1)))

    gabor_images = []
    for wavelength in wavelengths:
        sigma = slratio * wavelength
        accumulated_img = np.zeros_like(image, dtype=np.float32)

        for orientation_idx in range(number_of_orientations):
            orientation = (np.pi / number_of_orientations) * orientation_idx
            kernel = gabor_kernel(wavelength, orientation, sigma, aspect_ratio)
            filtered_img = cv2.filter2D(image, cv2.CV_32F, kernel)
            accumulated_img = np.maximum(accumulated_img, filtered_img, accumulated_img)

        # reduced_img = reduce_central_ring(accumulated_img)
        gabor_images.append(accumulated_img)

    return gabor_images

def automatic_threshold(image):
    # Calculate histogram of pixel intensities
    histogram, _ = np.histogram(image.flatten(), bins=np.arange(257))

    # Calculate the ratio of each intensity
    image_size = image.size
    ratios = histogram / image_size

    # Calculate the product of each intensity and its ratio and sum them up
    threshold_value = np.sum(np.arange(len(ratios)) * ratios) + 15.5

    #print(threshold_value)

    return threshold_value

def apply_threshold(gabor_images):
    binary_images = []
    for lol in gabor_images:
        # Apply Laplace filter
        laplacian_img = cv2.Laplacian(lol, cv2.CV_32F)

        # Add the Laplacian image to the original Gabor image
        combined_img = cv2.add(lol, laplacian_img)

        # Determine threshold value using the automatic thresholding method
        threshold_value = automatic_threshold(combined_img)

        # Apply threshold
        _, binary_img = cv2.threshold(combined_img, threshold_value, 255, cv2.THRESH_BINARY)

        binary_images.append(binary_img)

    return binary_images