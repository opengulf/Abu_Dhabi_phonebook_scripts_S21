files="$1"/*.png
# echo $files
# echo "${#files[@]}"

arr=($files)
len=${#arr[@]}
echo "Processing $len files."

i=1
for entry in "$1"/*.png; do
    # echo $entry
    filename=$(basename -- "$entry")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo "$filename.$extension $i/$len."
    #   # echo "$extension"
    #   # echo "$filename"
    #   tesseract $entry "$2/$filename" --psm 11 hocr
    # testvar = $2/$filename-border.$extension
    convert -bordercolor white -border 20 $entry $2/$filename-border.$extension
    ((i++))
done

# add_border() {
#     local entry=$1
#     filename=$(basename -- "$entry")
#     extension="${filename##*.}"
#     filename="${filename%.*}"
#     echo "$filename.$extension"
#     convert -bordercolor white -border 20 $entry $2/$filename-border.$extension
# }

# N=4

# for entry in "$1"/*.png; do
#     ((i = i % N))
#     ((i++ == 0)) && wait
#     add_border "$entry" &
# done
