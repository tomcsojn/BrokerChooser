#!/usr/bin/env bash

# Make sure you've built your Docker image (e.g. translator2) before running this script:
#   docker build -t translator2 .

declare -a translations=(
  "Helsinki-NLP/opus-mt-en-es es"
  "Helsinki-NLP/opus-mt-en-fr fr"
  "Helsinki-NLP/opus-mt-en-de de"
  "Helsinki-NLP/opus-mt-en-jap jap"
  "Helsinki-NLP/opus-mt-en-ar ar"
  "Helsinki-NLP/opus-mt-en-hi hi"
  "Helsinki-NLP/opus-mt-tc-big-en-pt por"
)

for t in "${translations[@]}"; do
  model=$(echo "$t" | cut -d ' ' -f1)
  lang=$(echo "$t" | cut -d ' ' -f2)
  echo "Translating with model=$model and language=$lang ..."
  # Invoke Docker with GPU support and input/output CSV files
  docker run --rm -v "/${PWD}:/app" --gpus all translator2 \
    --input_csv translated_output.csv \
    --output_csv "translated_${lang}.csv" \
    --language "$lang" \
    --model "$model"
  echo
done