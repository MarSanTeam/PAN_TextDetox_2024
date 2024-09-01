import csv
import json
import re

import pandas as pd
from datasets import load_dataset


def del_toxic_phrases(given_dataset):
    """
    Delete toxic phrases in datasets

    Args:
        given_dataset (Dataset): the database

    Returns:
        Dataframe
    """
    dfs = []
    # Iterate over languages
    for lang in given_dataset.keys():
        lang_dataset = given_dataset[lang]
        lang_texts = lang_dataset['text']
        lang_df = pd.DataFrame({'Text': lang_texts, 'Language': lang})
        dfs.append(lang_df)
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def convert_tsv_to_json(tsv_file_path):
    """
    Convert a TSV file to a list of dictionaries and return it.

    Args:
        tsv_file_path (str): The path to the TSV file.

    Returns:
        List[dict]: A list of dictionaries containing the data from the TSV file.
    """
    json_data = []
    with open(tsv_file_path, 'r') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            json_data.append(row)
    return json_data


def save_tsv_to_json(tsv_file_path, json_file_path):
    """
    Convert a TSV file to JSON format and save it to a JSON file.

    Args:
        tsv_file_path (str): The path to the TSV file.
        json_file_path (str): The path to save the JSON file.
    """
    data = []
    with open(tsv_file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)

    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


def remove_toxic_samples(text, given_dataset):
    """
    Remove toxic words from the input text using the provided pattern.

    Args:
        text (str): The input text containing toxic words.
        given_dataset (Dataset): The compiled regular expression pattern to match toxic words.

    Returns:
        Tuple[str, List[str]]: A tuple containing the cleaned text and a list of toxic words removed.
    """
    # Load the dataset
    dataset = load_dataset(given_dataset)

    # Initialize an empty list to aggregate all toxic texts
    all_toxic_texts = []

    # Loop through each split and language
    for split_name, split_dataset in dataset.items():
        for example in split_dataset:
            all_toxic_texts.append(example["text"])

    # Create a regular expression pattern to match words in the text
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, all_toxic_texts)) + r')\b')
    # Find all toxic words in the text
    toxic_words = pattern.findall(text)
    # Remove toxic words from the text
    cleaned_text = pattern.sub('', text)
    return cleaned_text, toxic_words


if __name__ == '__main__':
    dataset_name = "textdetox/multilingual_toxic_lexicon"
    # Example usage
    input_text = "Your ass text here fucked"
    clean_text, toxic_word = remove_toxic_samples(input_text, dataset_name)

    # Use the cleaned_text for further processing or analysis
    print("Cleaned text:", clean_text)
    print("Toxic words removed:", toxic_word)
