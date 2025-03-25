from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
import argparse
import pandas as pd

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, target_language, model_name):
    tokenizer, model = load_model(model_name)
    text = f"translate English to {target_language}: {text}"

    # inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    # translated = model.generate(**inputs)
    # translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    translator = pipeline(f"translation_en_to_{target_language}", model=model,PreTrainedTokenizer=tokenizer)
    translated_text=translator(text)[0]['translation_text']
    return translated_text

def translate_csv(input_csv, target_language, output_csv, model_name):
    df = pd.read_csv(input_csv)
    if 'english' not in df.columns:
        raise ValueError("Input CSV must contain an 'english' column for English text.")
    df[f'{target_language}'] = df['english'].apply(lambda x: translate_text(x, target_language, model_name))
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
    
    if args.text:
        print(f"Translated Text: {translate_text(args.text, args.language, args.model)}")
    
    if args.input_csv:
        translate_csv(args.input_csv, args.language, args.output_csv, args.model)

if __name__ == "__main__":
    main()
