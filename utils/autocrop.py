"""
This script is used to crop dataset images to a certain dimension from the center.

This helps in cases where you want to identify small objects in an image by magnifying them. This also helps 
in circumstances where you want to reduce the data size of images for faster training.

Dev Note:
The purpose of this script to the developer's use case is that the dataset consists of images with more
than 1000px in width and height, which may have contributed to the high memory usage (more than 12GB). 

Since the features are not too complex and the use case is highly specific, cropping from the center helped 
speed up training time and reduce memory usage.
"""

import cv2
import argparse
from pathlib import Path

def crop(img, height, width):
    img_h, img_w = img.shape[0], img.shape[1]

    crop_h = height if height < img_h else img_h
    crop_w = width if width < img_w else img_w

    mid_x, mid_y = int(img_w/2), int(img_h/2)
    cw2, ch2 = int(crop_w/2), int(crop_h/2)

    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path_to_dir = Path(args.path)

    if not path_to_dir.exists():
        print("Directory not found.")
        raise SystemExit(1)

    for file_entry in path_to_dir.iterdir():
        valid_fileformats = ['jpeg', 'jpg', 'png']
        fn_tokens = str(file_entry).split('.')

        if len(fn_tokens) == 2:
            filename, fileformat = fn_tokens[0], fn_tokens[1]

            if fileformat in valid_fileformats:
                image = cv2.imread(str(file_entry))
                crop_img = crop(image, 1500, 1500)
                cv2.imwrite("{}_cropped.{}".format(filename, fileformat), crop_img)