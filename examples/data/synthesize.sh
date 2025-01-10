#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE="synthetic/$(basename "$INPUT_FILE")"

mkdir -p synthetic
./preview.py "$INPUT_FILE"
metasyn fileformat "$INPUT_FILE" /tmp/ff.json
metasyn create-meta "$INPUT_FILE" -o /tmp/gmf.json
# metasyn synthesize /tmp/gmf.json --preview
metasyn synthesize /tmp/gmf.json --filehandler /tmp/ff.json -o "$OUTPUT_FILE"
./preview.py "$OUTPUT_FILE"
