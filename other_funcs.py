import numpy as np

def replace_black_ring(im_enh, im_mask):
    row, col = im_mask.shape
    area_sum = np.zeros((50, 50))

    posit = np.ceil((np.random.rand(3, 2) + 1) * 1 / 3 * min(row, col)).astype(int)

    for i in range(3):
        x = posit[i, 0]
        y = posit[i, 1]
        area_rand = im_enh[x - 25:x + 25, y - 25:y + 25]  # Select the background
        area_sum = area_sum + area_rand

    area_sum = area_sum * 1 / 3

    mean_val = np.mean(area_sum)  # Calculate the mean of the array
    mean_mask = np.logical_not(im_mask) * mean_val  # Generate a new background
    im_new = mean_mask + im_enh * im_mask  # Overlay background

    return im_new, mean_val

