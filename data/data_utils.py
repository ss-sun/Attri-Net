import numpy as np

def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):
    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.
    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o
    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o
    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)
    if (max_i - min_i) != 0:
        image = (np.divide((image - min_i), (max_i - min_i)) * (max_o - min_o) + min_o).copy()
        image[image > max_o] = max_o
        image[image < min_o] = min_o
    else:
        image = image
    return image

def normalize_image(img):
    img = np.float32(np.array(img))
    m = np.mean(img)
    s = np.std(img)
    if s == 0:
        s = 1e-06
    img = np.divide((img - m), s)
    return img
