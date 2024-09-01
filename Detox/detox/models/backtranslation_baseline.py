#!/usr/bin/env python3

import argparse
import logging
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
from transformers import (M2M100ForConditionalGeneration, NllbTokenizerFast, BartTokenizerFast, T5TokenizerFast,
                          BartForConditionalGeneration, T5ForConditionalGeneration, PreTrainedTokenizerFast,
                          PreTrainedModel, )


def get_model(type: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Returns a pre-trained model and tokenizer based on the specified type.

    Args:
        type (str): The type of model to retrieve.
        Valid options are "translator", "en_detoxifier", and "ru_detoxifier".

    Returns:
        (PreTrainedModel, PreTrainedTokenizer)
    Raises:
        ValueError: If an invalid type choice is provided.

    Examples:
        model, tokenizer = get_model("translator")
        model, tokenizer = get_model("en_detoxifier")
        model, tokenizer = get_model("ru_detoxifier")
    """
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model_types: Dict[str, Tuple[str, PreTrainedModel, PreTrainedTokenizerFast]] = {
        "translator": ("/mnt/disk2/LanguageModels/facebook_nllb", M2M100ForConditionalGeneration, NllbTokenizerFast,) }

    if type not in model_types:
        raise ValueError("Invalid type choice")

    model_name, ModelClass, TokenizerClass = model_types[type]

    logging.info(f"Loading {type} model: {model_name}")

    model = ModelClass.from_pretrained(model_name).eval().to(device)
    tokenizer = TokenizerClass.from_pretrained(model_name)

    return model, tokenizer


def translate_batch(texts: List[str], model: M2M100ForConditionalGeneration, tokenizer: PreTrainedTokenizerFast,
                    batch_size: int = 32, ) -> List[str]:
    """
    Translate a batch of texts.

    Args:
        texts (List[str]): The list of texts to translate.
        model (M2M100ForConditionalGeneration): The translation model.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the translation model.
        batch_size (int, optional): The batch size for translation. Defaults to 32.

    Returns:
        List[str]: The translated texts.
    """
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i: i + batch_size]
        batch_translated = model.generate(
            **tokenizer(batch, return_tensors="pt", padding=True, truncation=True, ).to(model.device),
            forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang], )
        translations.extend(tokenizer.decode(tokens, skip_special_tokens=True) for tokens in batch_translated)
    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtranslation baseline for PAN 2024 text detoxification task"
                                                 "that performs detox: translate input (toxic) text from "
                                                 "source language into pivot language (English), detox it "
                                                 "and then translate detoxified text back into source language")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for translation and detoxification.", )
    parser.add_argument("--input", type=str,
                        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_test(1).tsv",
                        help="Path to the input CSV file containing text data.", )
    parser.add_argument("--output", type=str,
                        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_test_en.tsv",
                        help="Path to the output CSV file to write the translated data.", )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    lang_id_mapping = {"ru": "rus_Cyrl", "en": "eng_Latn", "am": "amh_Ethi", "es": "spa_Latn", "uk": "ukr_Cyrl",
                       "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva", "de": "deu_Latn", }

    # Read input CSV
    input_df = pd.read_csv(args.input, sep='\t')

    # Filter non-English texts
    non_english_texts_df = input_df[input_df["lang"] != "en"]

    # Translate non-English texts
    model, tokenizer = get_model("translator")
    tokenizer.src_lang = lang_id_mapping[input_df["lang"].iloc[0]]
    tokenizer.tgt_lang = lang_id_mapping["en"]
    translated_texts = translate_batch(non_english_texts_df["toxic_sentence"].tolist(), model, tokenizer, args.batch_size)

    # Add translated texts as new columns
    input_df.loc[input_df["lang"] != "en", "translated_text"] = translated_texts
    input_df.loc[input_df["lang"] != "en", "translated_lang"] = "en"

    # Write to output CSV
    input_df.to_csv(args.input,sep = "\t",  index=False)


if __name__ == "__main__":
    import pandas as pd

    raw_test_file = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_test(1).tsv"

    main()
