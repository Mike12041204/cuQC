#!/bin/bash

# Check if the number of command-line arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <graph_file>"
    exit 1
fi

chmod 600 "$1"

python3 edgeToAdj.py "$1" 1 >temp.txt

python3 remove_duplicates.py temp.txt "$1" 

python3 rmEmptyVert.py "$1" "$1"

rm temp.txt
