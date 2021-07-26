files="$1"/*
# echo $files
# echo "${#files[@]}"

arr=($files)
len=${#arr[@]}
echo "Processing $len files."


for entry in "$1"/*; do
  filename=$(basename -- "$entry")
  extension="${filename##*.}"
  filename="${filename%.*}"
  echo "$filename.$extension $i/$len."
  # echo "$extension"
  # echo "$filename"
  tesseract $entry "$2/$filename" --psm 11 hocr
  ((i++))
done
