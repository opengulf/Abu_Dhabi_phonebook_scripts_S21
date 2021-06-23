for entry in "$1"/*.png; do
    # echo $entry
    filename=$(basename -- "$entry")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo "$filename.$extension"
    #   # echo "$extension"
    #   # echo "$filename"
    #   tesseract $entry "$2/$filename" --psm 11 hocr
    # testvar = $2/$filename-border.$extension
    convert -bordercolor white -border 20 $entry $2/$filename-border.$extension
done
