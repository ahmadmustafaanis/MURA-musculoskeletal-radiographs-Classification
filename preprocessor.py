import cv2
import numpy as np


def find_imp_area_xray(img):
    try:

        gray = img.astype("uint8")
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape

        # make bottom 2 rows black where they are white the full width of the image
        thresh[hh - 3 : hh, 0:ww] = 0

        # get bounds of white pixels
        white = np.where(thresh == 255)
        xmin, ymin, xmax, ymax = (
            np.min(white[1]),
            np.min(white[0]),
            np.max(white[1]),
            np.max(white[0]),
        )

        # crop the image at the bounds adding back the two blackened rows at the bottom
        crop = img[ymin : ymax + 3, xmin:xmax, :]
        #     print(crop.shape)
        crop = cv2.resize(crop, (256, 256))

        crop = crop / 255.0  # Optional, check with this and without this
        crop = crop.reshape(crop.shape + (1,))
        return crop
    except Exception as E:
        print(E)
        return img
