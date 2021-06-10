import os
import argparse
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew

# image = io.imread('input.png')
# grayscale = rgb2gray(image)
# angle = determine_skew(grayscale)
# rotated = rotate(image, angle, resize=True) * 255
# io.imsave('output.png', rotated.astype(np.uint8))


def process_files(args):

    path = args.input
    out_path = args.output
    raw_imgs = []
    deskewed_imgs = []

    if os.path.isfile(path) and path.endswith(('.jpeg', '.png')):
        raw_imgs.append(("/" + os.path.basename(path)))
        cropped_jpeg_temp = "/" + \
            os.path.splitext(os.path.basename(path))[
                0] + "_deskewed" + os.path.splitext(path)[1]
        deskewed_imgs.append(cropped_jpeg_temp)
        print(raw_imgs)
        print(deskewed_imgs)
        path = os.path.dirname(path)
    else:
        for file in os.listdir(path):
            uncropped_jpeg_temp = ""
            cropped_jpeg_temp = ""
            if file.endswith(('.jpeg', '.png')):
                uncropped_jpeg_temp = "/" + file
                # print (uncropped_jpeg)
                cropped_jpeg_temp = os.path.splitext(
                    file)[0] + "_deskewed" + os.path.splitext(file)[1]
                raw_imgs.append(uncropped_jpeg_temp)
                deskewed_imgs.append(cropped_jpeg_temp)
                # print(cropped_jpeg)

    pg_count = 0
    for raw_img in raw_imgs:
        print("Processing: " + raw_img)
        print("-------------------------------")
        out = out_path + deskewed_imgs[pg_count]
        # orig_im = Image.open(path + uncropped_jpeg)

        image = io.imread(path + raw_img)
        grayscale = rgb2gray(image)
        angle = determine_skew(grayscale)
        rotated = rotate(image, angle, resize=True) * 255
        if len(rotated) < len(rotated[0]):
            new_rotated = rotate(rotated, 90, resize=True)
            io.imsave(out, new_rotated.astype(np.uint8))
        else:
            io.imsave(out, rotated.astype(np.uint8))

        pg_count += 1

        # text_im.save(out_path + cropped_jpeg_list[pg_count] + "-c" + str(
        #         i) + os.path.splitext(uncropped_jpeg_list[pg_count])[1])


def main():
    parser = argparse.ArgumentParser(
        description="Deskew images.")
    # parser.add_argument("-type", help="Select a type of image process, full or minimal",
    #                     dest="type", type=str, required=True)
    parser.add_argument("-in", help="Input file directory",
                        dest="input", type=str, required=True)
    parser.add_argument("-out", help="Output file directory",
                        dest="output", type=str, required=True)
    # parser.add_argument("-n", help="Number of sampled boxes.",
    #                     dest="n", type=int, required=True, default=10)
    parser.set_defaults(func=process_files)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
