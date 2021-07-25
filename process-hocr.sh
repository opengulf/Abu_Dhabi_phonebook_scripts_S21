echo "JSON ------------------------------- \n"
python3 hocr-process.py -in $1/hocr/ -out $1/json
echo "CLUSTER ------------------------------------------ \N"
python3 hocr-process.py -in $1/hocr/ -out $1/json-clustered/ -type cluster
echo "INDENT ------------------------------------------ \N"
python3 hocr-process.py -in $1/hocr/ -out $1/json-indented/ -type indent
echo "NICK ------------------------------------------ \N"
python3 py-hocr-detect-columns.py -build-image False -in $1/hocr/ -mode P -path-out $1/json-nick/