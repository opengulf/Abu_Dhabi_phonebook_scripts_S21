files="$1"/*.png
# echo $files
# echo "${#files[@]}"

arr=($files)
len=${#arr[@]}
echo "Processing $len files."

i=1
for entry in "$1"/*.png; do
    filename=$(basename -- "$entry")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo "$filename.$extension $i/$len."
    convert -bordercolor white -border 20 $entry $2/$filename-border.$extension
    ((i++))
done
