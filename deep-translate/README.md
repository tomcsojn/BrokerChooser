Build the Docker image:

docker build -t translator .

1. Translate a single text (use --text and --language):

docker run --rm translator --text "Hello World" --language "de"

2. Evaluate translations on a dataset (mount local CSV file):

docker run --rm -v "$(pwd):/app" translator --dataset translated_output.csv --language hu

docker run --rm -v "$(pwd):/app" translator --dataset translated_output.csv --language hu --translator gpt --apikey OPENAI_API_KEY

(use ${pwd} on windows system)


3. Translate a CSV to a new CSV:

docker run --rm -v "$(pwd):/app" translator --input_csv translated_output.csv --output_csv translated.csv --language fr
