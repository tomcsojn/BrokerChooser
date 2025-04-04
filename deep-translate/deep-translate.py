import argparse
import pandas as pd
from deep_translator import GoogleTranslator, ChatGptTranslator, MicrosoftTranslator, DeeplTranslator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

# Function to initialize the appropriate translator based on user input
def get_translator(translator, api_key=None, target=None):
    if translator == 'google':
        return GoogleTranslator(target=target)
    elif translator == 'gpt':
        return ChatGptTranslator(api_key=api_key, target=target)
    elif translator == 'microsoft':
        return MicrosoftTranslator(api_key=api_key, target=target)
    elif translator == 'deepl':
        return DeeplTranslator(api_key=api_key, source='en', target=target, use_free_api=True)

# Function to translate text while preserving placeholders (e.g., [PLACEHOLDER])
def translate_text(text, translator):
    # Extract placeholders from the text
    placeholders = re.findall(r"\[.*?\]", text)
    # Map placeholders to numbered placeholders (e.g., [1], [2])
    numbered_placeholders = {f"[{i+1}]": placeholder for i, placeholder in enumerate(placeholders)}
    temp_text = text
    # Replace original placeholders with numbered placeholders
    for i, placeholder in enumerate(placeholders):
        temp_text = temp_text.replace(placeholder, f"[{i+1}]")
    # Translate the text with placeholders replaced
    translated_text = translator.translate(temp_text)
    # Replace numbered placeholders back with the original placeholders
    for number, placeholder in numbered_placeholders.items():
        translated_text = translated_text.replace(number, placeholder)
    return translated_text

# Function to evaluate translations using BLEU score
def evaluate_translations(dataset_path, target_language, translator):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    # Ensure required columns are present
    if 'english' not in df.columns or 'translated_value' not in df.columns:
        raise ValueError("Dataset must contain 'english' and 'translated_value' columns.")
    smooth = SmoothingFunction().method4  # Smoothing function for BLEU score
    # Translate the English text and store in a new column
    df['new_translated_value'] = df['english'].apply(lambda x: translate_text(x, translator))
    # Calculate BLEU scores for each row
    bleu_scores = df.apply(
        lambda x: sentence_bleu(
            [x['new_translated_value'].split()], 
            x['translated_value'].split(), 
            smoothing_function=smooth
        ), 
        axis=1
    )
    # Calculate the average BLEU score
    avg_bleu_score = bleu_scores.mean()
    print(f"Average BLEU Score for {target_language}: {avg_bleu_score}")
    return df

# Function to translate an entire CSV file
def translate_csv(input_csv, output_csv, translator):
    # Load the input CSV
    df = pd.read_csv(input_csv)
    # Ensure the 'english' column is present
    if 'english' not in df.columns:
        raise ValueError("Input CSV must contain an 'english' column for English text.")
    # Translate the English text and store in a new column
    df["translated_value"] = df['english'].apply(lambda x: translate_text(x, translator))
    # Save the translated data to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")

# Main function to handle command-line arguments and execute the appropriate functionality
def main():
    parser = argparse.ArgumentParser(description='Translate text using LLM and evaluate translation quality.')
    # Command-line arguments
    parser.add_argument('--text', type=str, help='Input English text to translate')
    parser.add_argument('--language', type=str, help='Target language')
    parser.add_argument('--dataset', type=str, help='Path to the dataset for evaluation', default=None)
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV file for translation', default=None)
    parser.add_argument('--output_csv', type=str, help='Path to save the translated CSV file', default='translated_output.csv')
    parser.add_argument('--translator', type=str, choices=['google', 'gpt', 'microsoft', 'deepl'], help='Translator to use', default='google')
    parser.add_argument('--apikey', type=str, help='API key for the translator', default=None)
    args = parser.parse_args()
    
    # Initialize the translator
    translator = get_translator(args.translator, args.apikey, args.language)
    
    # Translate a single text input
    if args.text:
        print(f"Translated Text: {translate_text(args.text, translator)}")
    
    # Evaluate translations using a dataset
    if args.dataset:
        evaluate_translations(args.dataset, args.language, translator)
    
    # Translate an entire CSV file
    if args.input_csv:
        translate_csv(args.input_csv, args.output_csv, translator)

# Entry point of the script
if __name__ == "__main__":
    main()
