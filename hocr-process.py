from sys import path
from bs4 import BeautifulSoup
import numpy as np
import argparse
import os
import json
import math

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from sklearn import base, cluster
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import pyrr


def get_midpoint(obj):
    coords = obj['bbox']
    return [(coords[2] + coords[0]) / 2, (coords[3] + coords[1]) / 2]


def get_attribute_list(line, key):
    attr_list = [i.strip() for i in line['title'].split(';')
                 if i.strip().split(' ')[0] == key]
    # print(attr_list)
    return [float(i) for i in attr_list[0].split(" ")[1:]]


def sort_lines(lines):
    lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))

    return np.array(lines)


def intersect(origin, dir, bounds):

    invdir = [1 / dir[0], 1 / dir[1] if dir[1] != 0 else math.inf]
    sign = [(invdir[0] < 0), (invdir[1] < 0)]

    print(bounds)

    tmin = (bounds[sign[0]][0]) * invdir[0]
    tmax = (bounds[1-sign[0]][0]) * invdir[0]

    tymin = (bounds[sign[1]][1] - origin[1]) * invdir[1]
    tymax = (bounds[1-sign[1]][1] - origin[1]) * invdir[1]

    if (tmin > tymax) or (tymin > tmax):
        print("False")
        return False

    print("True")
    return True


def check_intersection(line, box):
    print(line)
    print(box)
    baseline = line["baseline"]
    line_bbox = line["bbox"]

    bbox = box["bbox"]
    # Getting slope and constant for parametric equation.
    m, b = baseline

    x = line_bbox[0]
    y = line_bbox[3]

 # Changing origin: (b' = y' + b + mx')
    O = [0, y + b - m * x]
    D = [1, m]

    print(f"O: {O}")
    print(f"D: {D}")

    # ray = pyrr.ray.create(O, D)
    # print(ray)

    # tmin = bbox[0]
    # tmax = bbox[2]

    # if D[1] != 0:
    #     tymin = (bbox[1] - O[1]) / D[1]
    #     tymax = (bbox[3] - O[1]) / D[1]
    # else:
    #     print(f"O[1]: {O[1]}")
    #     return bbox[1] <= O[1] and bbox[3] >= O[1]

    #     tymin = np.sign(bbox[1] - O[1]) * math.inf
    #     tymax = np.sign(bbox[3] - O[1]) * math.inf

    # if(tymin > tymax):
    #     temp = tymin
    #     tymin = tymax
    #     tymax = temp

    # print(f"Tmin: {tmin}, Tmax: {tmax}")
    # print(f"Tymin: {tymin}, Tymax: {tymax}")

    # return tmin <= tymin and tmax >= tymax
    bounds = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]

    return intersect(O, D, bounds)

    # b += line_bbox[3]

    print(baseline)
    print(bbox)
    print("-----------")


def load_hocr_lines(filepath, min_conf):
    '''Loads hocr into an array with relevant features.'''
    page_array = []
    rawhtml = BeautifulSoup(open(filepath, encoding='utf-8'), "lxml")
    line = None
    for element in rawhtml.html.body.div.find_all('span'):
        # page_array.append(line)
        if element['class'][0] == 'ocr_line':
            # If a line has been loaded before and it has words.
            if line is not None and line['words']:
                page_array.append(line)

            line = {'words': []}
            bbox = [int(i)
                    for i in element['title'].split(';')[0].split(' ')[1:]]
            baseline = get_attribute_list(element, 'baseline')

            # print(baseline)
            line['bbox'] = bbox
            line['baseline'] = baseline

        elif element['class'][0] == "ocrx_word":
            confidence = get_attribute_list(element, "x_wconf")[0]
            # print(confidence)
            if confidence > min_conf:
                word = {'string': element.string}
                bbox = [int(i)
                        for i in element['title'].split(';')[0].split(' ')[1:]]
                x_conf = int(element['title'].split(';')[1].split(' ')[-1])

                word['bbox'] = bbox
                word['x_conf'] = x_conf
                # print(bbox)

                line['words'].append(word)

            # else:
            #     print("Word Removed: " + line.string)

            #     line_list.append(int(line['id'].split('_')[2]))
            #     line_list += [int(i)
            #                   for i in line['title'].split(';')[0].split(' ')[1:]]
            #     line_list += [0, 0, 0]
    return np.array(page_array), rawhtml


def find_indented_lines(lines):
    lines = sort_lines(lines)

    line_dist = []

    for i in range(len(lines)-1):
        line1 = lines[i]
        line2 = lines[i+1]

        dist = abs(line2['bbox'][1] - line1['bbox'][1])

        line_dist.append(dist)

    line_dist = np.array(line_dist)
    median, std = np.median(line_dist), np.std(line_dist)
    mean = np.mean(line_dist)

    print(line_dist)
    print(median)
    print(std)

    # cut_off = std * 3
    # lower, upper = mean - cut_off, mean + cut_off
    # print(lower)
    # identify outliers
    # outliers = [line for line in data if x < lower or x > upper]
    # print('Identified outliers: %d' % len(outliers))

    q25, q75 = np.percentile(line_dist, 25), np.percentile(line_dist, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 3
    lower, upper = q25 - cut_off, q75 + cut_off
    print(lower)

    joined_lines = []

    joined_line = lines[0]

    for i, dist in enumerate(line_dist):
        if(dist <= 10):
            print(f"Distance: {dist}")
            print(lines[i])
            print(lines[i+1])
            print("-----------------------")
            joined_line = join_lines(joined_line, lines[i+1])
        else:
            joined_lines.append(joined_line)
            joined_line = lines[i+1]

    # print(indented_lines)

    # for line in indented_lines:
    #     print(line)

    # print(sorted_lines)

    # check_intersection(sorted_lines[0], sorted_lines[1])

    # for line in lines:
    #     for box in lines:
    #         if line != box:
    #             result = check_intersection(line, box)
    #             print(result)
    #             print("--------------")

    column_data = [line['bbox'][0] for line in joined_lines]  # Midpoints
    np_data = np.array(column_data)
    print(np_data)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np_data.reshape(-1, 1))
    print(kmeans.labels_)

    columns = [[], []]

    for i, cluster in enumerate(kmeans.labels_):
        columns[cluster].append(lines[i])

    # print(np.array(columns))

    for column in columns:
        for line in column:
            print(line)

        print("-----------------------------------------------")

    return joined_lines


def join_lines(line1, line2):

    cluster = {"words": []}
    cluster["words"].extend(line1["words"])
    cluster["words"].extend(line2["words"])
    bbox = union_crops(line1["bbox"], line2["bbox"])
    cluster["bbox"] = bbox

    return cluster


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return [min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)]


def cluster_lines_NN(lines, indent):
    '''Attempts to join name and phone number using kmeans to find columns and NN to find matches.'''

    column_data = [(line['bbox'][2] + line['bbox'][0]) /
                   2 for line in lines]  # Midpoints
    np_data = np.array(column_data)
    # print(np_data)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np_data.reshape(-1, 1))
    # print(kmeans.labels_)

    columns = [[], []]

    for i, cluster in enumerate(kmeans.labels_):
        columns[cluster].append(lines[i])

# TODO: Confirm that right col is actually a number.
# TODO: Sort columns
# TODO: Directly remove very low confidence lines. Maybe with avg of word confidence?

    columns.sort(key=lambda c: c[0]['bbox'])

    if indent:
        print("Finding indented lines----------------------------------------")
        columns[0] = find_indented_lines(columns[0])
        print("INDENTING")

    # print(columns[0])

# Use y values, rather than midpoint
    left_col = np.array([col['bbox'][3] for col in columns[0]])
    right_col = np.array([col['bbox'][3] for col in columns[1]])

    # print(left_col)
    # print(right_col)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(
        (right_col.reshape(-1, 1)))
    distances, indices = nbrs.kneighbors(left_col.reshape(-1, 1))

    # print(indices)

    clustered_lines = []

    for i, index in enumerate(indices):

        entry = columns[0][i]
        number = columns[1][index[0]]

        # cluster = {"words": []}
        # cluster["words"].extend(entry["words"])
        # cluster["words"].extend(number["words"])
        # bbox = union_crops(entry["bbox"], number["bbox"])
        # cluster["bbox"] = bbox

        cluster = join_lines(entry, number)

        clustered_lines.append(cluster)

        # print(cluster)

        # print(columns[1][index[0]])
        # print(columns[0][i])
        # print("--------------------------------")

    return clustered_lines

#     X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# >>> nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

    # for col in columns:
    #     print("---------------------------------------------------------------------")
    #     for line in col:
    #         print(line)


def cluster_lines_hierarchical(lines):

    data = [[1, line['bbox'][1]] for line in lines]

    np_data = np.array(data)
    print(np_data)

    # # clustering
    thresh = 15
    clusters = hcluster.fclusterdata(np_data, thresh, criterion="distance")

    print(clusters)

    clustered_data = [None] * (max(clusters) + 1)

    print(len(clustered_data))

    for i in range(len(data)):
        cluster_num = clusters[i]
        print(cluster_num)
        if clustered_data[cluster_num] is None:
            clustered_data[cluster_num] = []

        clustered_data[cluster_num].append(lines[i])

    return clustered_data

    # # plotting
    # plt.scatter(*np.transpose(data), c=clusters)
    # plt.axis("equal")
    # title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    # plt.title(title)
    # plt.show()


def dump_text(lines):
    text = ""
    for line in lines:
        # print(line)
        for word in line["words"]:
            text += (word["string"] + " ").rstrip("\n")
            # print(word)
        text += ("\n")

    return text


def process_hocr(args):

    # Gets files or file to process.
    if os.path.isdir(args.path):
        hocr_files = [file for file in os.listdir(
            args.path) if file.endswith('.hocr')]

        path = args.path
    elif args.path.endswith('.hocr'):
        print("File")
        hocr_files = [os.path.basename(args.path)]
        path = os.path.dirname(args.path)
    else:
        print("No .hocr files specified.")
        exit(0)

    min_conf = args.min_conf

    for file in hocr_files:
        full_file = os.path.join(path, file)
        print(file)

        array, raw = load_hocr_lines(full_file, min_conf)

        if args.type == "full":

            orig_im = Image.open(args.image)
            draw = ImageDraw.Draw(orig_im)

            for line in array:
                bbox = line['bbox']
                print(line['bbox'])
                baseline = line['baseline']
                print(baseline)
                # draw.rectangle(bbox, outline='purple', width=3)
                point1 = [bbox[0], bbox[3] + baseline[1]]
                point2 = [bbox[0] + bbox[2],
                          (bbox[2]*baseline[0] + bbox[3] + baseline[1])]
                points = point1 + point2
                # point1.extend(point2)
                # points = [(0, baseline[1]), (300, 300*baseline[0] + baseline[1])]
                print(points)
                draw.line(points, fill="red", width=2)
                # print(item['words'])
                for word in line['words']:
                    print(word)
            orig_im.show()

        clustered_data = array.tolist()

        if args.type == "cluster" or args.type == "indent":

            clustered_data = cluster_lines_NN(array, args.type == "indent")

        filename = os.path.splitext(os.path.basename(file))[0]

        print(filename)

        # print(clustered_data)

        outfile = open(os.path.join(args.out, filename + ".json"), 'w')
        outfile.write(json.dumps(clustered_data))
        outfile.close()

        outfile = open(os.path.join(args.out, filename + ".txt"), 'w')
        outfile.write(dump_text(clustered_data))
        outfile.close()

        # for cluster in clustered_data:
        #     outfile.write(str(cluster))

        # outfile.write(str(clustered_data))

        # clustered_data = cluster_lines_hierarchical(array)
        # for cluster in clustered_data:
        #     if cluster is not None:
        #         for item in cluster:
        #             # print(item)
        #             print(item['bbox'])
        #             # print(item['words'])
        #             for word in item['words']:
        #                 print(word)
        #     print("---------------------")

        # print(array)
        # print(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Parse hocr files and return entries")
    parser.add_argument("-in", help="Full-path directory containing hocr files",
                        dest="path", type=str, required=True)
    parser.add_argument("-out", help="Full-path directory for output.",
                        dest="out", type=str, required=True)
    parser.add_argument("-image", help="Image",
                        dest="image", type=str, required=False)
    parser.add_argument("-type", help="Type of execution.",
                        dest="type", type=str, required=False, choices=["full", "parse", "cluster", "indent"])

    parser.add_argument("-min-confidence", help="Set word confidence threshold.",
                        dest="min_conf", type=int, required=False, default=10)
    # parser.add_argument("-build-image", help="Set whether to make images (True/False)",
    #                     dest="make_image", default="False", type=str, required=True)
    # parser.add_argument("-jpegs", help="Name of directory (not path) containing jpegs",
    #                     dest="jpeg_directory", type=str, required=False)
    # parser.add_argument("-bbox-out", help="Full path to directory to place output bbox images",
    #                     dest="bbox_location", type=str, required=False)
    # parser.add_argument("-mode", help="Either (P)rint out extracted entries, apply (CRF-print) and print out entries, or (CRF) and save JSON entries in labeled-json directory",
    #                     dest="mode", type=str, required=True)
    # parser.add_argument("-path-training", help="Path to the training files for CRF classifer",
    #                     dest="crf_training_path", type=str, required=False)
    # parser.add_argument("-build-tsv", help="(False) or path to directory where tsv will be made",
    #                     dest="tsv_path", type=str, required=False)
    parser.set_defaults(func=process_hocr)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
