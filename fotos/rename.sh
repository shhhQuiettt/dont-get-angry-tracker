#!/bin/bash

dir=$1

if [ -z $dir ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi
i=1

for file in $dir/*; do
    # if jpg or png rename
    if [ -f $file ]; then
        ext=${file##*.}
        if [ $ext == "jpg" ] || [ $ext == "png" ] || [ $ext == "jpeg" ] ; then
            mv $file $dir/calibration$i.$ext
            i=$((i+1))
        fi
    fi
done
