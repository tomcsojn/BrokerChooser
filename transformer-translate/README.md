Build the Docker image:

docker build -t translator2 .

1. Translate a single text (use --text and --language):

docker run --rm translator --text "Hello World" --language "de"


3. Translate a CSV to a new CSV:

docker run --rm -v "$(pwd):/app" translator2 --input_csv translated_output.csv --output_csv translated.csv --language fr --model google/madlad400-3b-mt