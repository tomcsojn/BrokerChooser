# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 19:53:06 2025

@author: tomcs
"""
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction




def compare_translations(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Check for required columns
    for df, name in [(df1, 'csv1'), (df2, 'csv2')]:
        if 'english' not in df.columns or 'translated_value' not in df.columns:
            raise ValueError(f"{name} must contain 'english' and 'translated_value' columns.")

    # Merge on 'english' so rows match
    merged = pd.merge(
        df1[['english', 'translated_value']].rename(columns={'translated_value': 'ref'}),
        df2[['english', 'translated_value']].rename(columns={'translated_value': 'hyp'}),
        on='english', how='inner'
    )

    # Compute BLEU per row
    smooth = SmoothingFunction().method4
    merged['bleu_score'] = merged.apply(
        lambda x: sentence_bleu(
            [x['ref'].split()], 
            x['hyp'].split(),
            # smoothing_function=smooth
        ), axis=1
    )

    avg_bleu_score = merged['bleu_score'].mean()
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")
    
if __name__ == "__main__":
    compare_translations("deep-translate/translated_output.csv", "deep-translate/llm_translated.csv")

# df1 = pd.read_csv("deep-translate/translated_output.csv")
# df1.drop("translated_value",inplace=True,axis=1)
# df1.to_csv("english.csv",index=False)
