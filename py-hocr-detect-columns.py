import numpy as np
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from math import sqrt
from statistics import mean
from PIL import Image, ImageOps, ImageDraw
import re
import json
import argparse
import os
import uuid
from cdparser import Classifier, Features, LabeledEntry, Utils
import sys


def build_manifest(main_path, entries_json):
    directory_uuid = entries_json[0]['directory_uuid']
    page_uuid = entries_json[0]['page_uuid']
    num_columns = max([int(entries_json[i]['col']) for i in entries_json])
    num_skipped_lines = len(
        [i for i in entries_json if entries_json[i]['skipped_line_after'] == '1'])
    if not os.path.exists(os.path.join(main_path, 'manifest')):
        os.makedirs(os.path.join(main_path, 'manifest'))
    manifest_path = os.path.join(
        main_path, 'manifest', directory_uuid + '_manifest.txt')
    with open(manifest_path, 'a') as f:
        f.write('directory_uuid:' + directory_uuid + '\n')
        f.write('page_uuid:' + page_uuid + '\n')
        f.write('number_extracted_entries:' + str(len(entries_json)) + '\n')
        f.write('total_number_lines_hocr:' +
                entries_json[0]['total_lines_from_hocr'] + '\n')
        f.write('number_extracted_columns:' + str(num_columns) + '\n')
        f.write('number_skipped_lines:' + str(num_skipped_lines))
        f.write('\n\n')
    f.close()


def make_tsv(filepath, type):
    with open(filepath, 'w+') as f:
        f.write('\t'.join(['directory_uuid', 'page_uuid',
                'entry_uuid', type + '_count', 'offset_count', 'token']))
        f.write('\n')
        f.close()


def build_entries_tsv(entries_json, dir_tsv, directory_uuid):
    if not os.path.exists(os.path.join(dir_tsv, directory_uuid + '_subjects.tsv')):
        make_tsv(os.path.join(dir_tsv, directory_uuid +
                 '_subjects.tsv'), 'subject')
    with open(os.path.join(dir_tsv, directory_uuid + '_subjects.tsv'), 'a') as f:
        for rec in entries_json:
            subject_count = 0
            for subject in entries_json[rec]['labeled_entry']['subjects']:
                offset_count = 0
                for sub_token in subject.split():
                    f.write(entries_json[rec]['directory_uuid'] + '\t'
                            + entries_json[rec]['page_uuid'] + '\t'
                            + entries_json[rec]['entry_uuid'] + '\t')
                    f.write(str(subject_count) + '\t')
                    f.write(str(offset_count) + '\t')
                    f.write(sub_token + '\n')
                    offset_count += 1
                subject_count += 1
    f.close()
    if not os.path.exists(os.path.join(dir_tsv, directory_uuid + '_occupations.tsv')):
        make_tsv(os.path.join(dir_tsv, directory_uuid +
                 '_occupations.tsv'), 'occupation')
    with open(os.path.join(dir_tsv, directory_uuid + '_occupations.tsv'), 'a') as f:
        for rec in entries_json:
            occupation_count = 0
            for occupation in entries_json[rec]['labeled_entry']['occupations']:
                offset_count = 0
                for occ_token in occupation.split():
                    f.write(entries_json[rec]['directory_uuid'] + '\t'
                            + entries_json[rec]['page_uuid'] + '\t'
                            + entries_json[rec]['entry_uuid'] + '\t')
                    f.write(str(occupation_count) + '\t')
                    f.write(str(offset_count) + '\t')
                    f.write(occ_token + '\n')
                    offset_count += 1
                occupation_count += 1
    f.close()
    if not os.path.exists(os.path.join(dir_tsv, directory_uuid + '_locations.tsv')):
        make_tsv(os.path.join(dir_tsv, directory_uuid +
                 '_locations.tsv'), 'location')
    with open(os.path.join(dir_tsv, directory_uuid + '_locations.tsv'), 'a') as f:
        for rec in entries_json:
            location_count = 0
            for location in entries_json[rec]['labeled_entry']['locations']:
                offset_count = 0
                for loc_token in location['value'].split():
                    f.write(entries_json[rec]['directory_uuid'] + '\t'
                            + entries_json[rec]['page_uuid'] + '\t'
                            + entries_json[rec]['entry_uuid'] + '\t')
                    f.write(str(location_count) + '\t')
                    f.write(str(offset_count) + '\t')
                    f.write(loc_token + '\n')
                    offset_count += 1
                location_count += 1
    f.close()


def imagebuilder(r, col_locations, image_filename, std1, gap_locations, page_uuid, output_directory):

    # It would appear that the incoming cropped jpegs are grayscale, necessitating a conversion to RGB to move forward
    # For the overlay, we use RGBA to enable opacity settings.

    pageimg = Image.open(image_filename).convert('RGB')
    overlay = ImageDraw.Draw(pageimg, 'RGBA')

    color = {0: (255, 0, 0, 90), 1: (0, 0, 255, 90), 2: (0, 255, 0, 90)}

    # Draw the hocr boxes
    for i in range(len(r)):
        overlay.polygon([(r[i:i+1, 1], r[i:i+1, 4]),
                         (r[i:i+1, 1], r[i:i+1, 2]),
                         (r[i:i+1, 3], r[i:i+1, 2]),
                         (r[i:i+1, 3], r[i:i+1, 4])],
                        fill=color[int(r[i:i+1, 5])], outline=color[int(r[i:i+1, 5])])

    # Draw the column bounds

    for loc in col_locations:

        overlay.polygon([(loc - std1, 0), (loc + std1, 0),
                         (loc + std1, pageimg.size[1]),
                         (loc - std1, pageimg.size[1])],
                        fill=(255, 1, 255, 100))
        overlay.line([(loc, 0), (loc, pageimg.size[1])],
                     fill=(0, 0, 0, 127), width=4)

    # Draw the gap lines

    for gap_location in gap_locations:
        overlay.line([(0, gap_location), (pageimg.size[0], gap_location)],
                     fill=(0, 0, 0, 127), width=4)
    pageimg.save(os.path.join(output_directory, page_uuid + '.jpeg'), 'JPEG')


def json_from_hocr(line_array, page_html, page_uuid, directory_uuid):
    hocr_entries = {}
    entries_json = {}
    total_page_line_count = 0
    for line in page_html.html.body.div.find_all('span'):
        if line['class'][0] == 'ocr_line':
            total_page_line_count += 1
            id_num = int(line['id'].split('_')[2])
            words = ' '.join([word.string.replace('\n', '').strip()
                             for word in line.children])
            hocr_entries[id_num] = normalize_entry(words)
    entry_id = 0
    for keep_line in line_array:
        entry_uuid = uuid.uuid1()
        if keep_line[5] == 0:
            entries_json[entry_id] = {
                'directory_uuid': directory_uuid,
                'page_uuid': page_uuid,
                'entry_uuid': entry_uuid.hex,
                'total_lines_from_hocr': str(total_page_line_count),
                'original_hocr_line_number': str(keep_line[0]),
                'bbox': ' '.join([str(val) for val in keep_line[1:5]]),
                'col': str(keep_line[6]),
                'appended': 'no',
                'skipped_line_after': str(keep_line[7]),
                'complete_entry': hocr_entries[keep_line[0]]
            }
            entry_id += 1

            # Indents
        else:
            try:
                if entries_json[entry_id - 1]['skipped_line_after'] != "1":
                    if entries_json[entry_id - 1]['complete_entry'][-1] == '-':
                        entries_json[entry_id -
                                     1]['complete_entry'] += hocr_entries[keep_line[0]]
                    else:
                        entries_json[entry_id - 1]['complete_entry'] += ' ' + \
                            hocr_entries[keep_line[0]]
                    entries_json[entry_id - 1]['appended'] = 'yes'
                else:
                    entries_json[entry_id] = {
                        'directory_uuid': directory_uuid,
                        'page_uuid': page_uuid,
                        'entry_uuid': entry_uuid.hex,
                        'total_lines_from_hocr': str(total_page_line_count),
                        'original_hocr_line_number': str(keep_line[0]),
                        'bbox': ' '.join([str(val) for val in keep_line[1:5]]),
                        'col': str(keep_line[6]),
                        'appended': 'no',
                        'skipped_line_after': str(keep_line[7]),
                        'complete_entry': hocr_entries[keep_line[0]]
                    }
                    entry_id += 1
            except:

                # Cases where an indent is the first line in page array and there is no preceding entry

                entries_json[entry_id] = {
                    'directory_uuid': directory_uuid,
                    'page_uuid': page_uuid,
                    'entry_uuid': entry_uuid.hex,
                    'total_lines_from_hocr': str(total_page_line_count),
                    'original_hocr_line_number': str(keep_line[0]),
                    'bbox': ' '.join([str(val) for val in keep_line[1:5]]),
                    'col': str(keep_line[6]),
                    'appended': 'no',
                    'skipped_line_after': str(keep_line[7]),
                    'complete_entry': hocr_entries[keep_line[0]]
                }
                entry_id += 1

    return entries_json


def load_hocr_lines(filepath):
    page_array = []
    rawhtml = BeautifulSoup(open(filepath, encoding='utf-8'), "lxml")
    for line in rawhtml.html.body.div.find_all('span'):
        line_list = []
        if line['class'][0] == 'ocr_line':
            line_list.append(int(line['id'].split('_')[2]))
            line_list += [int(i)
                          for i in line['title'].split(';')[0].split(' ')[1:]]
            line_list += [0, 0, 0]
            page_array.append(line_list)
    return np.array(page_array), rawhtml


def normalize_entry(entry):
    replacements = [("‘", "'"), ("’", "'"), (" ay.", " av."),
                    (" ay,", " av,"), ("- ", "-"), (" -", "-"), ("\t", ' ')]
    for swap in replacements:
        entry = entry.replace(swap[0], swap[1])
    return ' '.join(entry.split())


def remove_precede_space(entry):
    if re.search(r'\s\.|\s\,', entry):
        entry = entry.replace(' .', '.').replace(' ,', ',')
    return entry


def normalize_labeled_entry(labeled_entry_dict):
    new_subjects = []
    for subject in labeled_entry_dict['subjects']:
        new_subjects.append(remove_precede_space(subject))
    labeled_entry_dict['subjects'] = new_subjects
    new_occs = []
    for occ in labeled_entry_dict['occupations']:
        new_occs.append(remove_precede_space(occ))
    labeled_entry_dict['occupations'] = new_occs
    new_locations = []
    for loc_dict in labeled_entry_dict['locations']:
        new_loc_dict = {}
        new_loc_dict['value'] = remove_precede_space(loc_dict['value'])
        try:
            new_loc_dict['labels'] = loc_dict['labels']
        except:
            pass
        new_locations.append(new_loc_dict)
    labeled_entry_dict['locations'] = new_locations
    return labeled_entry_dict


def build_entries(args):
    """
    Page Array Structure
    col 0 = ID number of line
    col 1-4 = bbbox x1, y1, x2, y2
    col 5 = 0:col line; 1: indent line; 2:kill line
    col 6 = 1:col 1; 2: col2 (column assignment)
    col 7 = 0:ok to append, 1: do not append (likely gap)"""

    root = '/'.join(args.path.split('/')[:-1])
    directory_uuid = root.split('/')[-1]
    hocr_files = [file for file in os.listdir(
        args.path) if file.endswith('.hocr')]

    print("Processing: ", directory_uuid)

    for hocr_file in hocr_files:
        page_uuid = hocr_file.replace('_rotated', '').replace(
            '_cropped', '').replace('.hocr', '')
        # try:
        raw_hocr_array, page_html = load_hocr_lines(
            os.path.join(args.path, hocr_file))

        # jpeg_path = os.path.join(
        #     root, args.jpeg_directory, hocr_file.replace('.hocr', '.jpeg'))

        ##
        #   Find our likely column locations
        ##

        kmeans = KMeans(n_clusters=8).fit(
            raw_hocr_array[:, 1].reshape(-1, 1))
        centroids = kmeans.cluster_centers_
        cands_cols = {}

        for j in range(len(centroids)):
            cands_cols[centroids[j, 0]] = 0
            std = sqrt(mean([((i - centroids[j, 0])**2)
                             for i in raw_hocr_array[:, 1]]))
            for i in range(len(raw_hocr_array)):
                if abs(raw_hocr_array[i:i+1, 1] - centroids[j, 0]) > (std/16):
                    pass
                else:
                    cands_cols[centroids[j, 0]] += 1

        # We have our dict of possible column location along with the number of entries proximate to each
        # But in case our k-means clustering was compromised by a slanted column line (yielding double col locations)
        # we need to check for that and take an average of the resultant double col1 and double col2 locations

        halfway_page_rough = (max(raw_hocr_array[:, 1])/2)*.95
        lefthand_cands = [
            i for i in cands_cols.items() if i[0] < halfway_page_rough]
        righthand_cands = [
            i for i in cands_cols.items()if i[0] > halfway_page_rough]
        col1_xval_cands = sorted(
            lefthand_cands, key=lambda x: x[1], reverse=True)
        col2_xval_cands = sorted(
            righthand_cands, key=lambda x: x[1], reverse=True)

        # Now that we've split our candidate locations into what we think is roughly the col1 and col2 areas
        # we look for two closely proximate top candidates. If they exist, we average them out

        if abs(col1_xval_cands[0][0] - col1_xval_cands[1][0]) < 50:
            col1_xval = mean(
                [col1_xval_cands[0][0], col1_xval_cands[1][0]])
        else:
            col1_xval = col1_xval_cands[0][0]
        try:
            if abs(col2_xval_cands[0][0] - col2_xval_cands[1][0]) < 50:
                col2_xval = mean(
                    [col2_xval_cands[0][0], col2_xval_cands[1][0]])
            else:
                col2_xval = col2_xval_cands[0][0]

        # For cases where we don't have multiple candidates for a second column,
        # possibly because Tesseract missed second column entirely:
        except:
            col2_xval = col2_xval_cands[0][0]

        ##
        #  Pass to identify and keep lines that are at x-val of column edges
        ##

        std1 = sqrt(mean([((i - col1_xval)**2)
                    for i in raw_hocr_array[:, 1]]))/16
        std2 = sqrt(mean([((i - col2_xval)**2)
                    for i in raw_hocr_array[:, 1]]))/16
        for i in range(len(raw_hocr_array)):
            if abs(raw_hocr_array[i:i + 1, 1] - col1_xval) < std1 or abs(raw_hocr_array[i:i + 1, 1] - col2_xval) < std2:
                # Col 1 identification:
                if abs(raw_hocr_array[i:i + 1, 1] - col1_xval) < std1:
                    raw_hocr_array[i:i + 1, 6] = 1
                # Col 2 identification:
                elif abs(raw_hocr_array[i:i + 1, 1] - col2_xval) < std2:
                    raw_hocr_array[i:i + 1, 6] = 2
            # Id of indents and any potential chopped off lines to right of columns
            elif raw_hocr_array[i:i + 1, 1] - (col1_xval + std1) > 0 and raw_hocr_array[i:i + 1, 1] < col2_xval:
                # Col 1 indents identification
                raw_hocr_array[i:i + 1, 5] = 1
                raw_hocr_array[i:i + 1, 6] = 1
            elif raw_hocr_array[i:i + 1, 1] - (col2_xval + std2) > 0:
                # Col 2 indents identification
                raw_hocr_array[i:i + 1, 5] = 1
                raw_hocr_array[i:i + 1, 6] = 2
            # Eliminating anything to left of column 1
            if raw_hocr_array[i:i+1, 1] < (col1_xval - std1):
                raw_hocr_array[i:i + 1, 5] = 2
                # Eliminating anything to right of column 2 right edge
            if raw_hocr_array[i:i + 1, 1] > (col2_xval + (col2_xval - col1_xval)):
                raw_hocr_array[i:i + 1, 5] = 2

        ##
        #   Pass to find lines flush with a column whose y vals make them unlikely to be in the page block for entries
        ##

        reduced_array = raw_hocr_array[raw_hocr_array[:, 5] != 2]
        sorted_y_array = np.sort(reduced_array.view(
            'i8,i8,i8,i8,i8,i8,i8,i8'), order=['f2'], axis=0).view(np.int)

        # To find an appropriate vertical line density of all likely lines, we grab a sample at a point roughly 1/4
        # the way through our entries, then gather the line density at that point within a gap around that point
        # The gap is calculated at 0.5% of the highest yval (roughly 0.5% of the yval height of the page)

        quarter_page = len(sorted_y_array)//4
        gap = float(max(raw_hocr_array[:, 2]))*.05
        entry_density = len([i for i in sorted_y_array[:, 2] if abs(
            i - sorted_y_array[quarter_page:quarter_page+1, 2]) < gap/2])

        # We now examine the line density around every line in the page; if the density is low, we do a second check to make
        # sure the reason isn't that it is a first or last line; in those cases we check for gap density after/before the line
        # Anything that still fails we cut

        for i in range(len(sorted_y_array)):
            proximate_lines = [yval for yval in sorted_y_array[:, 2] if abs(
                yval - sorted_y_array[i:i+1, 2]) < gap/2]
            if len(proximate_lines) - 1 < entry_density/2:
                top_line_proximate_lines = [yval for yval in sorted_y_array[:, 2] if yval - sorted_y_array[i:i + 1, 2] < gap
                                            and yval - sorted_y_array[i:i + 1, 2] > 0]
                bottom_line_proximate_lines = [yval for yval in sorted_y_array[:, 2] if
                                               sorted_y_array[i:i + 1, 2] - yval < gap and sorted_y_array[i:i + 1, 2] - yval > 0]
                if len(top_line_proximate_lines) > entry_density or len(bottom_line_proximate_lines) > entry_density:
                    pass
                else:
                    sorted_y_array[i:i+1, 5] = 2

        ##
        #   Pass to look for missing lines by finding no line with a yval that is 1.95% of the expected space
        #   between lines; if a missing line is found, we mark the previous line to make sure an indent isn't appended to that line
        #   Those indents following a gap will become a standalone line because their head-entry is missing
        ##

        line_only_array = sorted_y_array[sorted_y_array[:, 5] != 2]
        sorted_line_only_array = np.sort(line_only_array.view(
            'i8,i8,i8,i8,i8,i8,i8,i8'), order=['f6', 'f2'], axis=0).view(np.int)
        sample_lines = sorted_line_only_array[sorted_line_only_array[:, 6] == 1]
        gaps = []

        for i in range(len(sample_lines)):
            try:
                gaps.append(
                    int(sample_lines[i+1:i+2, 2] - sample_lines[i:i+1, 2]))
            except:
                pass

        average_line_gap = sum(gaps) // len(sample_lines)
        gap_locations = []
        for i in range(len(sorted_line_only_array)):
            try:
                if int(sorted_line_only_array[i + 1:i + 2, 2] - sorted_line_only_array[i:i + 1, 2]) > average_line_gap*1.95:
                    sorted_line_only_array[i:i+1, 7] = 1
                    gap_locations.append(
                        sorted_line_only_array[i:i + 1, 2] + average_line_gap*1.5)
            except:
                pass

        # We can either build the image or return the json

        if args.make_image == 'True':
            imagebuilder(sorted_line_only_array, [
                col1_xval, col2_xval], jpeg_path, std1, gap_locations, page_uuid, os.path.join(root, args.bbox_location))
        entries_json = json_from_hocr(
            sorted_line_only_array, page_html, page_uuid, directory_uuid)
        build_manifest(root, entries_json)
        if args.mode == 'P':
            print(entries_json)
            outpath = os.path.join(args.path_out, os.path.splitext(hocr_file)[0] + ".json")
            f = open(outpath, "w")
            f.write(json.dumps(entries_json))
            f.close()
            print("----------------------------------------")
        else:
            classifier = Classifier.Classifier()
            classifier.load_training(args.crf_training_path)
            classifier.train()
            for rec in entries_json:
                entry = LabeledEntry.LabeledEntry(
                    entries_json[rec]['complete_entry'])
                classifier.label(entry)
                final_entries = normalize_labeled_entry(entry.categories)
                entries_json[rec]['labeled_entry'] = final_entries
                if args.mode == 'CRF-print':
                    print(entries_json[rec])
            if args.mode == 'CRF':
                with open(os.path.join(root, 'final-entries', page_uuid + '_labeled.json'), 'w') as f:
                    for rec in sorted(entries_json.keys()):
                        f.write(json.dumps(entries_json[rec]) + '\n')
                f.close()
            if args.tsv_path != "False":
                build_entries_tsv(
                    entries_json, args.tsv_path, directory_uuid)
        print("Completed processing of ", page_uuid)

        # except Exception as exception:
        #     print(exception.__traceback__)
        #     print("Likely ad or problematic hocr in :", page_uuid, ". Skipped.")


def main():
    parser = argparse.ArgumentParser(
        description="Parse hocr files and return entries")
    parser.add_argument("-in", help="Full-path directory containing hocr files",
                        dest="path", type=str, required=True)
    parser.add_argument("-build-image", help="Set whether to make images (True/False)",
                        dest="make_image", default="False", type=str, required=True)
    parser.add_argument("-jpegs", help="Name of directory (not path) containing jpegs",
                        dest="jpeg_directory", type=str, required=False)
    parser.add_argument("-bbox-out", help="Full path to directory to place output bbox images",
                        dest="bbox_location", type=str, required=False)
    parser.add_argument("-mode", help="Either (P)rint out extracted entries, apply (CRF-print) and print out entries, or (CRF) and save JSON entries in labeled-json directory",
                        dest="mode", type=str, required=True)
    parser.add_argument("-path-training", help="Path to the training files for CRF classifer",
                        dest="crf_training_path", type=str, required=False)
    parser.add_argument("-path-out", help="Path to the training files for CRF classifer",
                        dest="path_out", type=str, required=False)
    parser.add_argument("-build-tsv", help="(False) or path to directory where tsv will be made",
                        dest="tsv_path", type=str, required=False)
    parser.set_defaults(func=build_entries)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
