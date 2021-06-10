import argparse
import os
from scipy.ndimage.filters import rank_filter
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

type = ""


def deskew(im, save_path, direct, max_skew=25):
    file_base, file_ext = os.path.splitext(save_path)

    open_cv_image = np.array(im)
    if open_cv_image.ndim == 2:
        channels = 0
    else:
        channels = open_cv_image.shape[2]

    if direct == "Y":
        height, width = im.shape[:2]
        # print(height)
        # print(width)

        # Create a grayscale image and denoise it
        if channels != 0:
            im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)
        else:
            im_gs = cv2.fastNlMeansDenoising(im, h=3)

        print("De-noise ok.")
        # Create an inverted B&W copy using Otsu (automatic) thresholding
        im_bw = cv2.threshold(
            im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        print("Otsu ok.")

        # Detect lines in this image. Parameters here mostly arrived at by trial and error.
        # If the initial threshold is too high, then settle for a lower threshold value

        threshold = 200
        im_draw = im.copy()
        # while threshold >= 50:
        print("Threshold: " + str(threshold))
        im_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(im_pil)

        lines = cv2.HoughLinesP(
            im_bw, 1, np.pi / 180, threshold, None, minLineLength=width / 4, maxLineGap=width / 200)
        # Collect the angles of these lines (in radians)
        angles = []
        if lines is not None:
            # draw = ImageDraw.Draw(im)
            # print(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(im_draw, (x1, y1), (x2, y2), (0, 0, 255))
                geom = np.arctan2(y2 - y1, x2 - x1)
                # print(np.rad2deg(geom))
                
                if abs(geom) < np.deg2rad(max_skew):
                    draw.line((x1,y1,x2,y2))
                    angles.append(geom)


            # cv2.imshow('image', im_draw)
            # cv2.waitKey()
            im_pil.show()
        # angles = [angle for angle in angles if abs(
        #     angle) < np.deg2rad(max_skew)]

            angles.sort()

        else:
            print("No lines.")
            exit(0)

        for angle in angles:
            print(np.rad2deg(angle))

        print("angle mean")
        print(np.rad2deg(np.mean(angles)))
        # print(-np.rad2deg(np.median(angles)) - 90)


        if len(angles) < 0:
            # Insufficient data to deskew
            print("Insufficient data to deskew.")
            # cv2.imwrite(img=im, filename=file_base+file_ext)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #im_pil = Image.fromarray(im)
            #im_pil.save(save_directory + cropped_jpeg_list[pg_count])
            print("Cropped not image saved.")
            return im

        # try:
        #     lines = cv2.HoughLinesP(
        #         im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150)
        #     # Collect the angles of these lines (in radians)
        #     angles = []
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         geom = np.arctan2(y2 - y1, x2 - x1)
        #         # print(np.rad2deg(geom))
        #         angles.append(geom)
        # except:
        #     lines = cv2.HoughLinesP(
        #         im_bw, 1, np.pi / 180, 150, minLineLength=width / 12, maxLineGap=width / 150)
        #     # Collect the angles of these lines (in radians)
        #     angles = []
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         geom = np.arctan2(y2 - y1, x2 - x1)
        #         # print(np.rad2deg(geom))
        #         angles.append(geom)

        else:
            # Average the angles to a degree offset
            angle_deg = -np.rad2deg(np.mean(angles))
            print(angle_deg)

            # Rotate the image by the residual offset
            M = cv2.getRotationMatrix2D(
                (width / 2, height / 2), angle_deg, 1)
            im = cv2.warpAffine(im, M, (width, height),
                                borderMode=cv2.BORDER_REPLICATE)

            # Plot if a full run
            # Always save deskewed image
            if type == "full":
                plt.subplot(111), plt.imshow(im)
                plt.title('Deskewed Image'), plt.xticks([]), plt.yticks([])
                plt.show()
            cv2.imwrite(img=im, filename=file_base + "_rotated" + file_ext)
            print("Only de-skewed cropped image saved.")
            return im
    else:
        height, width = im.shape[:2]
        print(height)
        print(width)

        # Create a grayscale image and denoise it
        im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

        # Create an inverted B&W copy using Otsu (automatic) thresholding
        im_bw = cv2.threshold(
            im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Detect lines in this image. Parameters here mostly arrived at by trial and error.
        # If the initial threshold is too high, then settle for a lower threshold value
        try:
            lines = cv2.HoughLinesP(
                im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150)
            # Collect the angles of these lines (in radians)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                geom = np.arctan2(y2 - y1, x2 - x1)
                print(np.rad2deg(geom))
                angles.append(geom)
        except TypeError:
            lines = cv2.HoughLinesP(
                im_bw, 1, np.pi / 180, 150, minLineLength=width / 12, maxLineGap=width / 150)
            # Collect the angles of these lines (in radians)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                geom = np.arctan2(y2 - y1, x2 - x1)
                print(np.rad2deg(geom))
                angles.append(geom)
        except:
            print(
                "TypeError encountered with HoughLines. Check cropped image output. Only cropped image saved.")
            return

        angles = [angle for angle in angles if abs(
            angle) < np.deg2rad(max_skew)]

        if len(angles) < 5:
            # Insufficient data to deskew
            print(
                "Insufficient data to deskew. Cropped image might already be straight.")
            return im

        else:

            # Average the angles to a degree offset
            angle_deg = np.rad2deg(np.median(angles))

            # Rotate the image by the residual offset
            M = cv2.getRotationMatrix2D(
                (width / 2, height / 2), angle_deg, 1)
            im = cv2.warpAffine(im, M, (width, height),
                                borderMode=cv2.BORDER_REPLICATE)

            # Plot if a full run
            # Always save deskewed image
            if type == "full":
                plt.subplot(111), plt.imshow(im)
                plt.title('Deskewed Image'), plt.xticks([]), plt.yticks([])
                plt.show()
            cv2.imwrite(img=im, filename=file_base + "_rotated" + file_ext)
            print("Rotated cropped image saved")
            return im


def load_image(args):
    im_path = os.path.basename(args.input)
    orig_im = cv2.imread(args.input)
    type = args.type
    deskew(orig_im, args.output + im_path, 'Y')


def main():
    parser = argparse.ArgumentParser(
        description="Deskew directory image.")
    parser.add_argument("-type", help="Select a type of image process, full or minimal",
                        dest="type", type=str, required=True)
    parser.add_argument("-in", help="Input file directory",
                        dest="input", type=str, required=True)
    parser.add_argument("-out", help="Output file directory",
                        dest="output", type=str, required=True)
    parser.set_defaults(func=load_image)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
