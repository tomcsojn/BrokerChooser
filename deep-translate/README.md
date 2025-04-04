### Build the Docker Image

To build the Docker image, run the following command:

```bash
docker build -t translator .
```

### Usage

#### 1. Translate a Single Text

Use the `--text` and `--language` flags to translate a single text:

```bash
docker run --rm translator --text "Hello World" --language "de"
```

#### 2. Evaluate Translations on a Dataset

Mount a local CSV file and evaluate translations using the `--dataset` and `--language` flags:

```bash
docker run --rm -v "$(pwd):/app" translator --dataset translated_output.csv --language hu
```

To use a specific translator (e.g., GPT) and an API key:

```bash
docker run --rm -v "$(pwd):/app" translator --dataset translated_output.csv --language hu --translator gpt --apikey OPENAI_API_KEY
```

> **Note:** On Windows systems, use `${pwd}` instead of `$(pwd)`.

#### 3. Translate a CSV to a New CSV

Translate an input CSV file to a new output CSV file using the `--input_csv`, `--output_csv`, and `--language` flags:

```bash
docker run --rm -v "$(pwd):/app" translator --input_csv translated_output.csv --output_csv translated.csv --language fr
```

To use a specific translator (e.g., GPT) and an API key:

```bash
docker run --rm -v "$(pwd):/app" translator --input_csv translated_output.csv --output_csv translated.csv --language fr --translator gpt --apikey OPENAI_API_KEY
```