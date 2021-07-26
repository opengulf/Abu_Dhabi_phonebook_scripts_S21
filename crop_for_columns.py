#! /usr/bin/env python

import argparse

from matplotlib.pyplot import imshow


def process_image(args):

    import os

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    from scipy.ndimage.filters import rank_filter
    from sklearn.cluster import KMeans

    path = args.input
    out_path = args.output

    def deskew(im, save_directory, direct, max_skew=10):
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
            try:
                lines = cv2.HoughLinesP(
                    im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150)
                # Collect the angles of these lines (in radians)
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    geom = np.arctan2(y2 - y1, x2 - x1)
                    # print(np.rad2deg(geom))
                    angles.append(geom)
            except:
                lines = cv2.HoughLinesP(
                    im_bw, 1, np.pi / 180, 150, minLineLength=width / 12, maxLineGap=width / 150)
                # Collect the angles of these lines (in radians)
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    geom = np.arctan2(y2 - y1, x2 - x1)
                    # print(np.rad2deg(geom))
                    angles.append(geom)

            angles = [angle for angle in angles if abs(
                angle) < np.deg2rad(max_skew)]

            if len(angles) < 5:
                # Insufficient data to deskew
                print(
                    "Insufficient data to deskew. Cropped image might already be straight. Cropped image saved.")
                cv2.imwrite(img=im,
                            filename=save_directory + cropped_jpeg_list[pg_count])
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                #im_pil = Image.fromarray(im)
                #im_pil.save(save_directory + cropped_jpeg_list[pg_count])
                print("Cropped image saved.")
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
                if args.type == "full":
                    plt.subplot(111), plt.imshow(im)
                    plt.title('Deskewed Image'), plt.xticks([]), plt.yticks([])
                    plt.show()
                cropped_jpeg = cropped_jpeg_list[pg_count]
                cv2.imwrite(img=im,
                            filename=save_directory + cropped_jpeg[:-5] + "_rotated.jpeg")
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
                if args.type == "full":
                    plt.subplot(111), plt.imshow(im)
                    plt.title('Deskewed Image'), plt.xticks([]), plt.yticks([])
                    plt.show()
                cropped_jpeg = cropped_jpeg_list[pg_count]
                cv2.imwrite(img=im,
                            filename=save_directory + cropped_jpeg[:-5] + "_rotated.jpeg")
                print("Rotated cropped image saved")
                return im

    def dilate(ary, N, iterations):
        """Dilate using an NxN '+' sign shape. ary is np.uint8."""
        kernel = np.zeros((N, N), dtype=np.uint8)
        kernel[(N-1)//2, :] = 1
        dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

        kernel = np.zeros((N, N), dtype=np.uint8)
        kernel[:, (N-1)//2] = 1
        dilated_image = cv2.dilate(
            dilated_image, kernel, iterations=iterations)

        if args.type == "full":
            plt.subplot(111), plt.imshow(dilated_image, cmap='gray')
            plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
            plt.show()

        return dilated_image

    def find_components(edges, max_components=16):
        """Dilate the image until there are just a few connected components.
        Returns contours for these components."""
        # Perform increasingly aggressive dilation until there are just a few
        # connected components.
        count = 410
        dilation = 5
        n = 1
        while count > 400:
            n += 1
            dilated_image = dilate(edges, N=3, iterations=n)
    #         print(dilated_image.dtype)
            dilated_image = cv2.convertScaleAbs(dilated_image)
    #         print(dilated_image.dtype)
            contours, hierarchy = cv2.findContours(
                dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
            print(count)
        # print dilation
        # Image.fromarray(edges).show()
        #Image.fromarray(255 * dilated_image).show()
        return contours

    def props_for_contours(contours, ary):
        """Calculate bounding box & the number of set pixels for each contour."""
        c_info = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            c_im = np.zeros(ary.shape)
            cv2.drawContours(c_im, [c], 0, 255, -1)
            c_info.append({
                'x1': x,
                'y1': y,
                'x2': x + w - 1,
                'y2': y + h - 1,
                'sum': np.sum(ary * (c_im > 0))/255
            })
        return c_info

    def union_crops(crop1, crop2):
        """Union two (x1, y1, x2, y2) rects."""
        x11, y11, x21, y21 = crop1
        x12, y12, x22, y22 = crop2
        return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)

    def intersect_crops(crop1, crop2):
        x11, y11, x21, y21 = crop1
        x12, y12, x22, y22 = crop2
        return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)

    def crop_area(crop):
        x1, y1, x2, y2 = crop
        return max(0, x2 - x1) * max(0, y2 - y1)

    def find_border_components(contours, ary):
        borders = []
        area = ary.shape[0] * ary.shape[1]
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 0.5 * area:
                borders.append((i, x, y, x + w - 1, y + h - 1))
        return borders

    def angle_from_right(deg):
        return min(deg % 90, 90 - (deg % 90))

    def remove_border(contour, ary):
        """Remove everything outside a border contour."""
        # Use a rotated rectangle (should be a good approximation of a border).
        # If it's far from a right angle, it's probably two sides of a border and
        # we should use the bounding box instead.
        c_im = np.zeros(ary.shape)
        r = cv2.minAreaRect(contour)
        degs = r[2]
        if angle_from_right(degs) <= 10.0:
            # box = cv2.cv.BoxPoints(r)
            box = cv2.boxPoints(r)
            box = np.int0(box)
            cv2.drawContours(c_im, [box], 0, 255, -1)
            cv2.drawContours(c_im, [box], 0, 0, 4)
        else:
            x1, y1, x2, y2 = cv2.boundingRect(contour)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

        return np.minimum(c_im, ary)

    def find_optimal_components_subset(contours, edges):
        """Find a crop which strikes a good balance of coverage/compactness.
        Returns an (x1, y1, x2, y2) tuple.
        """
        c_info = props_for_contours(contours, edges)
        # c_info.sort(key=lambda x: -x['sum'])
        total = np.sum(edges) / 255
        area = edges.shape[0] * edges.shape[1]

# Sorting crops downwards by area.
        c_info.sort(key=lambda cr: crop_area(
            (cr['x1'], cr['y1'], cr['x2'], cr['y2'])), reverse=True)

# Getting biggest n crops.
        c_info = c_info[:args.n]

        c = c_info[0]
        del c_info[0]
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        crop = this_crop
        covered_sum = c['sum']

        while covered_sum < total:
            changed = False
            recall = 1.0 * covered_sum / total
            prec = 1 - 1.0 * crop_area(crop) / area
            f1 = 2 * (prec * recall / (prec + recall))
            # print '----'
            for i, c in enumerate(c_info):
                this_crop = c['x1'], c['y1'], c['x2'], c['y2']
                new_crop = union_crops(crop, this_crop)
                new_sum = covered_sum + c['sum']
                new_recall = 1.0 * new_sum / total
                new_prec = 1 - 1.0 * crop_area(new_crop) / area
                new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

                # Add this crop if it improves f1 score,
                # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
                # ^^^ very ad-hoc! make this smoother
                remaining_frac = c['sum'] / (total - covered_sum)
                new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
                if new_f1 > f1 or (remaining_frac > 0.25 and new_area_frac < 0.15):
                    print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                        i, covered_sum, new_sum, total, remaining_frac,
                        crop_area(crop), crop_area(
                            new_crop), area, new_area_frac,
                        f1, new_f1))
                    crop = new_crop
                    covered_sum = new_sum
                    del c_info[i]
                    changed = True
                    break

            if not changed:
                break

        return crop

    def pad_crop(crop, contours, edges, border_contour, pad_px=15):
        """Slightly expand the crop to get full contours.
        This will expand to include any contours it currently intersects, but will
        not expand past a border.
        """
        bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
        if border_contour is not None and len(border_contour) > 0:
            c = props_for_contours([border_contour], edges)[0]
            bx1, by1, bx2, by2 = c['x1'] + \
                5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

        def crop_in_border(crop):
            x1, y1, x2, y2 = crop
            x1 = max(x1 - pad_px, bx1)
            y1 = max(y1 - pad_px, by1)
            x2 = min(x2 + pad_px, bx2)
            y2 = min(y2 + pad_px, by2)
            return crop

        crop = crop_in_border(crop)

        c_info = props_for_contours(contours, edges)

        c_info.sort(key=lambda cr: crop_area(
            (cr['x1'], cr['y1'], cr['x2'], cr['y2'])), reverse=True)

        c_info = c_info[:args.n]

        changed = False
        for c in c_info:
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            this_area = crop_area(this_crop)
            int_area = crop_area(intersect_crops(crop, this_crop))
            new_crop = crop_in_border(union_crops(crop, this_crop))
            if 0 < int_area < this_area and crop != new_crop:
                print('%s -> %s' % (str(crop), str(new_crop)))
                changed = True
                crop = new_crop

        if changed:
            return pad_crop(crop, contours, edges, border_contour, pad_px)
        else:
            return crop

    def downscale_image(im, max_dim=2048):
        """Shrink im until its longest dimension is <= max_dim.
        Returns new_image, scale (where scale <= 1).
        """
        a, b = im.size
        if max(a, b) <= max_dim:
            return 1.0, im

        scale = 1.0 * max_dim / max(a, b)
        new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
        return scale, new_im

    def find_outliers(columns):
        col_median = np.median(columns, axis=0)

        col_widths = [col[2] - col[0] for col in columns]
        width_median = np.median(col_widths)
        print(col_widths)
        print(width_median)

        outliers = []
        print(col_median)

        for i, col in enumerate(columns):
            np_col = np.array(col)
            diff = abs(col_median - np_col)
            rates = diff / col_median

            width_diff = abs(width_median - col_widths[i])

            crop_outliers = list(map((lambda rate: rate > args.thresh), rates))
            crop_outliers[0] = False
            crop_outliers[2] = False

            if width_diff / width_median > args.thresh:
                print("Outlier in position: " + str(i))

                # Checking right boundary against column on the right.
                if i + 1 < len(columns):
                    next_col = columns[i+1]
                    if abs(next_col[0] - col[3]) / next_col[0] > args.thresh:
                        crop_outliers[3] = True
                        print("Outlier on right boundary")

                # Checking left boundary against col on left.
                if i > 0:
                    prev_col = columns[i-1]
                    if abs(col[0] - prev_col[3]) / prev_col[3] > args.thresh:
                        crop_outliers[0] = True
                        print("Outlier on left boundary")

                if not crop_outliers[0] and not crop_outliers[2]:
                    crop_outliers[0] = True
                    crop_outliers[2] = True

            outliers.append(crop_outliers)

            print(rates)
            print(crop_outliers)

        return outliers

    def correct_outliers(columns, outliers):
        # print(columns)
        columns = [list(col) for col in columns]
        print(columns)
        corrected_columns = columns.copy()

        top_data = [col for col, outlier in zip(
            columns, outliers) if not outlier[1]]

        bottom_data = [col for col, outlier in zip(
            columns, outliers) if not outlier[3]]

        left_data = []
        right_data = []

        for i, col in enumerate(columns):
            outlier = outliers[i]

            if not outlier[0]:
                data = [i, col[0]]
                print(data)
                left_data.append(data)

            if not outlier[2]:
                data = [i, col[2]]
                print(data)
                right_data.append(data)

        m1, b1 = np.polyfit([i[0] for i in top_data], [i[1]
                            for i in top_data], 1)

        m3, b3 = np.polyfit([i[2] for i in bottom_data], [i[3]
                            for i in bottom_data], 1)

        m0, b0 = np.polyfit([i[0] for i in left_data], [i[1]
                            for i in left_data], 1)

        m2, b2 = np.polyfit([i[0] for i in right_data], [i[1]
                            for i in right_data], 1)

        print(f"The equation of 0 is f(x)={m0}(x) + {b0}")
        print(left_data)

        print(f"The equation of 1 is f(x)={m1}(x) + {b1}")
        print(top_data)

        print(f"The equation of 2 is f(x)={m2}(x) + {b2}")
        print(right_data)

        for i, outlier in enumerate(outliers):
            col = columns[i]

            if outlier[1]:
                # If line is going upwards (up goes to 0)
                if m1 < 0:
                    col[1] = m1 * col[2] + b1
                else:
                    col[1] = m1 * col[0] + b1

            if outlier[3]:
                if m3 < 0:
                    col[3] = m3 * col[0] + b3
                else:
                    col[3] = m3 * col[2] + b3

            if outlier[0]:
                col[0] = m0 * i + b0

        return columns

    # Creates an empty list that takes on the filename of each jpeg in the directory
    # Then, it will loop through every single one of them
    uncropped_jpeg_list = []
    cropped_jpeg_list = []
    if os.path.isfile(path) and path.endswith(('.jpeg', '.png')):
        uncropped_jpeg_list.append(("/" + os.path.basename(path)))
        cropped_jpeg_temp = "/" + \
            os.path.splitext(os.path.basename(path))[0] + "_cropped"
        cropped_jpeg_list.append(cropped_jpeg_temp)
        print(uncropped_jpeg_list)
        print(cropped_jpeg_list)
        path = os.path.dirname(path)
    else:
        for file in os.listdir(path):
            uncropped_jpeg_temp = ""
            cropped_jpeg_temp = ""
            if file.endswith(('.jpeg', '.png')):
                uncropped_jpeg_temp = "/" + file
                # print (uncropped_jpeg)
                cropped_jpeg_temp = os.path.splitext(file)[0] + "_cropped"
                uncropped_jpeg_list.append(uncropped_jpeg_temp)
                cropped_jpeg_list.append(cropped_jpeg_temp)
                # print(cropped_jpeg)

    pg_count = 0
    total_pages = len(uncropped_jpeg_list)

    # For each image
    for uncropped_jpeg in uncropped_jpeg_list:
        print("Processing: " + uncropped_jpeg)
        print(f"File {pg_count}/{total_pages}")
        print("-------------------------------")

        # Downscaling
        orig_im = Image.open(path + uncropped_jpeg)
        scale, im = downscale_image(orig_im)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(np.asarray(im), kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        # Detect edge and plot
        edges = cv2.Canny(img, 100, args.canny)

        if args.type == "full":
            plt.subplot(111), plt.imshow(edges, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

            plt.show()

        # TODO: dilate image _before_ finding a border. This is crazy sensitive!
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Seems to find bounding boxes based on contours.
        borders = find_border_components(contours, edges)
        print(borders)
        # Sorts by area ascending
        if len(borders) > 1:
            borders.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        border_contour = None
        if len(borders):
            border_contour = contours[borders[0][0]]
            edges = remove_border(border_contour, edges)

        edges = 255 * (edges > 0).astype(np.uint8)

        # Remove ~1px borders using a rank filter.
        maxed_rows = rank_filter(edges, -4, size=(1, 20))
        maxed_cols = rank_filter(edges, -4, size=(20, 1))
        debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
        edges = debordered

        contours = find_components(edges)
        if len(contours) == 0:
            #        print '%s -> (no text!)' % path
            return

        # Gets crops based on contours
        c_info = props_for_contours(contours, edges)

        # Sorting by area descending and getting biggest n crops.
        c_info.sort(key=lambda cr: crop_area(
            (cr['x1'], cr['y1'], cr['x2'], cr['y2'])))
        c_info = c_info[-args.n:]

        c_info_clean = c_info.copy()
        centers = []
        # print(c_info)

        # Getting x-axis midpoint to classify column.
        for i in range(len(c_info)):
            c = c_info[i]
            center = ((c['x1'] + c['x2']) / 2)
            print(str(center) + " -> " + str(c))
            centers.append(center)

        centers_np = np.array(centers)
        # print(centers_np)

        # Running K-means to get four different columns.
        kmeans = KMeans(n_clusters=4, random_state=0).fit(
            centers_np.reshape(-1, 1))
        print(kmeans.labels_)

        # print(c_info_clean)

        colors = ['blue', 'green', 'yellow', 'brown']

        columns = [None] * 4

        draw = ImageDraw.Draw(im)

        # Drawing crops and aggregating crops per column.
        for i, c in enumerate(c_info_clean):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            col = kmeans.labels_[i]

            draw.rectangle(this_crop, outline=colors[col], width=2)

            if columns[col] is None:
                columns[col] = this_crop
            else:
                columns[col] = union_crops(columns[col], this_crop)

        # Sort columns from left to right
        columns.sort(key=lambda col: col[0])

        if args.correct:

            outliers = find_outliers(columns)

            try:
                corrected_columns = correct_outliers(columns, outliers)
                print(corrected_columns)
            except TypeError:
                print("Error in outlier detection. Too much variance.")
                print("Review file: " + uncropped_jpeg_list[pg_count])
                corrected_columns = columns

        else:
            corrected_columns = columns

        # Drawing final columns.
        if args.type == "full" or args.type == "border":
            for col in corrected_columns:
                draw.rectangle(col, outline='purple', width=3)
                print(col)
            im.show()

        # Saving columns.
        for i, col in enumerate(corrected_columns):
            upsized_crop = [int(x / scale) for x in col]
            text_im = orig_im.crop(upsized_crop)

            text_im.save(out_path + cropped_jpeg_list[pg_count] + "-c" + str(
                i) + os.path.splitext(uncropped_jpeg_list[pg_count])[1])

        pg_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Read a scanned street directory image, crop, and deskew.")
    parser.add_argument("-type", help="Select a type of image process, full or minimal",
                        dest="type", type=str, required=True)
    parser.add_argument("-in", help="Input file directory",
                        dest="input", type=str, required=True)
    parser.add_argument("-out", help="Output file directory",
                        dest="output", type=str, required=True)
    parser.add_argument("-n", help="Number of sampled boxes.",
                        dest="n", type=int, required=True, default=10)
    parser.add_argument("-correct", help="Number of sampled boxes.",
                        dest="correct", type=bool, required=False, default=False)
    parser.add_argument("-threshold", help="Threshold for outlier detection.",
                        dest="thresh", type=float, required=False, default=0.1)

    parser.add_argument("-canny", help="Threshold for Canny thresholding.",
                        dest="canny", type=float, required=False, default=400)
    parser.set_defaults(func=process_image)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
