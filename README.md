# Abu Dhabi Calling! OCR Scripts

This repo is an adaptation of Nicholas Wolf's scripts with some additional scripts. The original README can be found at the end of this file.

This takes single-page directory scans from the Abu Dhabi Calling! project collection and processes it into a .json file with potential entries for post-processing.

## Pipeline

1. Take raw single-page scans and add a white border around it. The purpose of this trick is to make column detection more accurate and consistent for tightly cropped pages. This is done with **batch-border.sh**.

2. Find and save crops of every column in the page. For this, run **crop_for_columns.py**.

3. Run tesseract on all crops and get hocr file with entries. Use **batch-tess.sh**.

4. Convert hocr into json, and remove low-confidence lines and words. Optionally, try to join indented entries using different methods with varying degrees of accuracy. Run **process-hocr.sh**.


## Example Run

### Preprocessing

1. Create directory structure needed for entire process using ```./create-folders.sh <out-dir>```. This command will create ```<out-dir>``` as well as the following basic structure.
```
<out-dir>
├── borders
├── crops
├── hocr
├── json
├── json-raw
├── json-clustered
├── json-indented
└── json-nick

```

2. Add a white border around all scans using ```batch-border.sh <scans-dir> <out-dir>/borders```.

3. Detect and extract all columns using the following command:
```
python3 crop_for_columns.py -in <out-dir>/borders/ -out <out-dir>/crops/ -n 50 -type border -correct True -thresh 0.5 -canny 200
```

These are the arguments:
+ **-in**: Input image directory.
+ **-out**: Output crops directory.
+ **-n**: Number of samples for column detection. It will use the _n_ largest rectangles by area to form all columns. This is a mechanism to remove noise.
+ **-type**: ```border|minimal```. If ```border``` is chosen, it will show the original image with all detected rectangles and the extracted columns. If ```minimal``` is chosen, it will process all files silently.
+ **-correct**: Tells the program whether or not to check for outlier column coordinates. Can be either ```True``` or  ```False```. **_Note:_** Since detection parameters have changed over time to be more robust, this option might not be necessary.

+ **-thresh**: Sets the threshold for outlier column values. This number is used to check whether the difference between a coordinate and the median of all coordinates for that point is above ```thresh%```.

+ **-canny**: Threshold for Canny edge detection algorithm. 200 seems to work consistently.

### Tesseract

4. Run all extracted columns through Tesseract to get the hocr files. Use ```./batch-tess.sh <out-dir>/crops/ <out-dir>/hocr/```. This script uses page segmentation method 11.

### Post-Processing

5. Convert and extract all information from hocr files. If all files are needed, run ```./process-hocr.sh <out-dir>```

# NYC Historical City Directories Support Scripts

**crop_for_columns.py**

Pre-processing of images to assist Tesseract as much as possible. Performs deskew (straight entry lines help not just Tesseract, but with discovery of column edges later), grayscale, and cropping of non-text areas using dilation to discover text boundaries on page.

**py-hocr-detect-columns.py**

After Tesseract is run on each page, producing an HOCR file for each page, this script parses the HOCR, turning each line into a value in a numpy array for ease of use. It detects columns using K-means clustering on bounding box x-vals, then indented lines for appending to previous line to make a complete entry. Y-vals are used to detect any remaining OCR capture outside of the desired text block, and also any skipped lines.

Once it has reconstructed each entry, the components of the entry are parsed into subject, occupation, address using Conditional Random Fields. The result is a labeled entry in a dictionary structure. These results are written out as an NDJSON file.

**address_final_transform.py**

Example of the post-processing procedure whereby a field, in this case the address field, is cleaned up using known swaps (see **example-corrections-swap-file.json** file; these swap files were generated out of successive clustering and hand-checking that outlier values should in fact be replaced with high-frequency correct tokens), then clustered using ngram fingerprint method for a basic dimension reduction, then clustered one last time using Levenshtein distance to assign a score of likelihood of correctness. Output are "scored" entries packaged in NDJSON files, one for each city directory.
