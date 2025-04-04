from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import argparse
import pandas as pd
import torch
import re

def load_model(model_name):
    # Load the tokenizer and model from the specified pre-trained model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Add special tokens for placeholders, e.g., [1], [2], ..., up to 20 placeholders
    additional_tokens = [f"[{i}]" for i in range(1, 21)]
    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to include new tokens
    return tokenizer, model

def translate_text(text, target_language, tokenizer, model):
    # Extract placeholders (e.g., [1], [2], ...) from the input text
    placeholders = re.findall(r"\[.*?\]", text)
    # Map numbered placeholders (e.g., [1]) to their original placeholders
    numbered_placeholders = {f"[{i+1}]": placeholder for i, placeholder in enumerate(placeholders)}
    
    # Replace original placeholders with numbered placeholders in the text
    temp_text = text
    for i, placeholder in enumerate(placeholders):
        temp_text = temp_text.replace(placeholder, f"[{i+1}]")
    
    # Create a translation pipeline using the specified model and tokenizer
    translator = pipeline(
        f"translation_en_to_{target_language}",  # Translation task
        model=model,
        tokenizer=tokenizer,
        max_length=400,  # Maximum length of the translated text
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
    )
    
    # Perform the translation
    translated_text = translator(temp_text)[0]['translation_text']
    
    # Replace numbered placeholders back with their original placeholders
    for number, placeholder in numbered_placeholders.items():
        translated_text = translated_text.replace(number, placeholder)
    
    return translated_text

def translate_csv(input_csv, target_language, output_csv, tokenizer, model):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure the CSV contains an 'english' column for translation
    if 'english' not in df.columns:
        raise ValueError("Input CSV must contain an 'english' column for English text.")
    
    # Translate each row in the 'english' column and store the results in a new column
    df['translated_value'] = df['english'].apply(lambda x: translate_text(x, target_language, tokenizer, model))
    
    # Save the translated DataFrame to the specified output CSV file
    df.to_csv(output_csv, index=False)
    print(f"Translated CSV saved to {output_csv}")

def main():
    # Set up argument parsing for command-line usage
    parser = argparse.ArgumentParser(description='Translate text using a Transformer-based model.')
    parser.add_argument('--text', type=str, help='Input English text to translate')  # Single text input
    parser.add_argument('--language', type=str, required=True, help='Target language name (e.g., fr, jap)')  # Target language
    parser.add_argument('--model', type=str, required=True, help='Model name to use for translation', default='google/madlad400-3b-mt')  # Model name
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV file for translation', default=None)  # Input CSV file
    parser.add_argument('--output_csv', type=str, help='Path to save the translated CSV file', default='translated_output.csv')  # Output CSV file
    args = parser.parse_args()
    
    # Load the tokenizer and model
    tokenizer, model = load_model(args.model)

    # Translate a single text input if provided
    if args.text:
        print("Translated Text:", translate_text(args.text, args.language, tokenizer, model))

    # Translate a CSV file if an input CSV path is provided
    if args.input_csv:
        translate_csv(args.input_csv, args.language, args.output_csv, tokenizer, model)

if __name__ == "__main__":
    main()
