Build the Docker image:

docker build -t translator2 .

1. Translate a single text (use --text and --language):

docker run --rm translator --gpus all --text "Hello World" --language "de"


3. Translate a CSV to a new CSV:

docker run --rm -v "$(pwd):/app" --gpus all translator2 --input_csv translated_output.csv --output_csv translated.csv --language fr --model google/madlad400-3b-mt