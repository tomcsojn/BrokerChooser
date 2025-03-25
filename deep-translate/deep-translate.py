import argparse
import pandas as pd
from deep_translator import GoogleTranslator,ChatGptTranslator,MicrosoftTranslator,DeeplTranslator
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import re

def get_translator(translator,api_key=None,target=None):
    if translator == 'google':
        return GoogleTranslator()
    elif translator == 'gpt':
        return ChatGptTranslator(api_key=api_key,target=target)
    elif translator == 'microsoft':
        return MicrosoftTranslator(api_key=api_key,target=target)
    elif translator == 'deepl':
        return DeeplTranslator(api_key=api_key,source='en',target=target,use_free_api=True)

def translate_text(text, translator):
    placeholders = re.findall(r"\[.*?\]", text)
    temp_text = re.sub(r"\[.*?\]", "TEMP_PLACEHOLDER", text)
    translated_text = translator.translate(temp_text)
    for placeholder in placeholders:
        translated_text = translated_text.replace("TEMP_PLACEHOLDER", placeholder, 1)
    return translated_text


def evaluate_translations(dataset_path, target_language, translator):
    df = pd.read_csv(dataset_path)
    smooth = SmoothingFunction().method4
    df['new_translated_value'] = df['english'].apply(lambda x: translate_text(x, translator))
    bleu_scores = df.apply(lambda x: sentence_bleu([x['translated_value'].split()], x['new_translated_value'].split(), smoothing_function=smooth), axis=1)
    avg_bleu_score = bleu_scores.mean()
    print(f"Average BLEU Score for {target_language}: {avg_bleu_score}")
    return df

def translate_csv(input_csv, target_language, output_csv, translator):
    df = pd.read_csv(input_csv)
    if 'english' not in df.columns:
        raise ValueError("Input CSV must contain an 'english' column for English text.")
    df[f'{target_language}'] = df['english'].apply(lambda x: translate_text(x, translator))
    df["translated_value"] = df['english'].apply(lambda x: translate_text(x, translator))
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Translate text using LLM and evaluate translation quality.')
    parser.add_argument('--text', type=str, help='Input English text to translate')
    parser.add_argument('--language', type=str, help='Target language')
    parser.add_argument('--dataset', type=str, help='Path to the dataset for evaluation', default=None)
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV file for translation', default=None)
    parser.add_argument('--output_csv', type=str, help='Path to save the translated CSV file', default='translated_output.csv')
    parser.add_argument('--translator', type=str, choices=['google', 'gpt', 'microsoft', 'deepl'], help='Translator to use', default='google')
    parser.add_argument('--apikey', type=str, help='API key for the translator', default=None)
    args = parser.parse_args()
    
    translator = get_translator(args.translator, args.apikey, args.language)
    
    if args.text:
        print(f"Translated Text: {translate_text(args.text, translator)}")
    
    if args.dataset:
        evaluate_translations(args.dataset, args.language,translator)
    
    if args.input_csv:
        translate_csv(args.input_csv, args.language, args.output_csv, translator)

if __name__ == "__main__":
    main()
