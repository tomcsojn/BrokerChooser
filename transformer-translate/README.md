### Build the Docker Image

To build the Docker image, run the following command:

```bash
docker build -t translator2 .
```

### Usage

#### 1. Translate a Single Text

To translate a single text, use the `--text` and `--language` options:

```bash
docker run --rm --gpus all translator2 --text "Hello World" --language "de"
```

#### 2. Translate a CSV File

To translate a CSV file and save the output to a new CSV file, use the following command:

```bash
docker run --rm -v "/$(pwd):/app" --gpus all translator2 --input_csv translated_output.csv --output_csv translated.csv --language fr --model google/madlad400-3b-mt
```
Use smaller models like opus

```bash
docker run --rm -v "/$(pwd):/app" --gpus all translator2 --input_csv translated_output.csv --output_csv translated_hu.csv --language hu --model Helsinki-NLP/opus-mt-en-hu
```


#### 3. Translate for All Languages with Small Models

To run translations for all languages using small models, execute the script:

```bash
bash ./run_opus_translation.sh
```