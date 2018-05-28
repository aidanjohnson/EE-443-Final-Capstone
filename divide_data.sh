#!/bin/bash

path=$1
dest=$2

num=$(ls -A1 $path | wc -l)
move=$(echo "scale=0;$num*0.3" | bc)

mapfile -t sample < <(shuf -n $move -e $path/*); mv "${sample[@]}" $dest