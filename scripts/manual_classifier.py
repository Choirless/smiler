import cv2
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--folder", required=True,
    help="path to folder containing images")
args = parser.parse_args()

extensions = ('.png','.jpg','.jpeg')

filenames = [file for file in os.listdir(args.folder) if file.lower().endswith(extensions)]

for filename in filenames:
    print("processing:", filename)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(f"{args.folder}/{filename}")

    cv2.imshow('Image', image)
    key = cv2.waitKey(5000)

    if key == ord('a'):
        print("happy")
        cv2.imwrite(f"happy/{filename}", image)

    if key == ord('l'):
        print("not happy")
        cv2.imwrite(f"nothappy/{filename}", image)


