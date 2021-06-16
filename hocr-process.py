from sys import path
from bs4 import BeautifulSoup
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from sklearn import base, cluster
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def get_midpoint(obj):
    coords = obj['bbox']
    return [(coords[2] + coords[0]) / 2, (coords[3] + coords[1]) / 2]


def get_attribute(line, key):
    attr_list = [i.strip() for i in line['title'].split(';')
                 if i.strip().split(' ')[0] == key]
    # print(attr_list)
    return [float(i) for i in attr_list[0].split(" ")[1:]]


def load_hocr_lines(filepath):
    '''Loads hocr into an array with relevant features.'''
    page_array = []
    rawhtml = BeautifulSoup(open(filepath, encoding='utf-8'), "lxml")
    word_list = None
    for line in rawhtml.html.body.div.find_all('span'):
        # page_array.append(line)
        if line['class'][0] == 'ocr_line':
            if word_list is not None:
                page_array.append(word_list)

            word_list = {'words': []}
            bbox = [int(i) for i in line['title'].split(';')[0].split(' ')[1:]]
            baseline = get_attribute(line, 'baseline')

            # print(baseline)
            word_list['bbox'] = bbox
            word_list['baseline'] = baseline

        elif line['class'][0] == "ocrx_word":
            word = {'string': line.string}
            bbox = [int(i) for i in line['title'].split(';')[0].split(' ')[1:]]
            x_conf = int(line['title'].split(';')[1].split(' ')[-1])

            word['bbox'] = bbox
            word['x_conf'] = x_conf
            # print(bbox)

            word_list['words'].append(word)

            #     line_list.append(int(line['id'].split('_')[2]))
            #     line_list += [int(i)
            #                   for i in line['title'].split(';')[0].split(' ')[1:]]
            #     line_list += [0, 0, 0]
    return np.array(page_array), rawhtml


def cluster_lines_NN(lines):
    '''Attempts to join name and phone number using kmeans to find columns and NN to find matches.'''

    column_data = [(line['bbox'][2] + line['bbox'][0]) /
                   2 for line in lines]  # Midpoints
    np_data = np.array(column_data)
    print(np_data)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np_data.reshape(-1, 1))
    print(kmeans.labels_)

    columns = [[], []]

    for i, cluster in enumerate(kmeans.labels_):
        columns[cluster].append(lines[i])

# TODO: Confirm that right col is actually a number.
# TODO: Sort columns
# TODO: Directly remove very low confidence lines. Maybe with avg of word confidence?

    columns.sort(key=lambda c: c[0]['bbox'])


# Use y values, rather than midpoint
    left_col = np.array([col['bbox'][1] for col in columns[0]])
    right_col = np.array([col['bbox'][1] for col in columns[1]])

    print(left_col)
    print(right_col)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
        (right_col.reshape(-1, 1)))
    distances, indices = nbrs.kneighbors(left_col.reshape(-1, 1))

    print(indices)

    for i, index in enumerate(indices):
        print(columns[1][index[0]])
        print(columns[0][i])
        print("--------------------------------")

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


def process_hocr(args):

    # Gets files or file to process.
    if os.path.isdir(args.path):
        hocr_files = [file for file in os.listdir(
            args.path) if file.endswith('.hocr')]
    elif args.path.endswith('.hocr'):
        hocr_files = [args.path]
    else:
        print("No .hocr files specified.")
        exit(0)

    for file in hocr_files:
        array, raw = load_hocr_lines(file)

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
        # clustered_data = cluster_lines_NN(array)

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
    parser.add_argument("-image", help="Image",
                        dest="image", type=str, required=False)
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
