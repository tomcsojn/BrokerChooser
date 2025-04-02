from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
import argparse
import pandas as pd
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, target_language, tokenizer, model):
    translator = pipeline(
        f"translation_en_to_{target_language}",
        model=model,
        tokenizer=tokenizer,
        max_length=400,
        device=0 if torch.cuda.is_available() else -1
    )
    translated_text = translator(text)[0]['translation_text']
    return translated_text
def translate_csv(input_csv, target_language, output_csv, tokenizer, model):
    df = pd.read_csv(input_csv)
    if 'english' not in df.columns:
        raise ValueError("Input CSV must contain an 'english' column for English text.")
    df['translated_value'] = df['english'].apply(lambda x: translate_text(x, target_language, tokenizer, model))
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Translate text using a Transformer-based model.')
    parser.add_argument('--text', type=str, help='Input English text to translate')
    parser.add_argument('--language', type=str, required=True, help='Target language name (e.g., Spanish, French)')
    parser.add_argument('--model', type=str, required=True, help='Model name to use for translation', default='google/madlad400-3b-mt')
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV file for translation', default=None)
    parser.add_argument('--output_csv', type=str, help='Path to save the translated CSV file', default='translated_output.csv')
    args = parser.parse_args()
    
    tokenizer, model = load_model(args.model)

    if args.text:
        print("Translated Text:", translate_text(args.text, args.language, tokenizer, model))

    if args.input_csv:
        translate_csv(args.input_csv, args.language, args.output_csv, tokenizer, model)

if __name__ == "__main__":
    main()
