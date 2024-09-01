# -*- coding: utf-8 -*-
import csv
import json
import re

from datasets import load_dataset

from detox.config import DETOX_INS_KEY, DETOX_END_KEY, DETOX_INSTRUCTION, DETOX_RES_KEY


def create_instruction(sample, mode):
    """
    Generate an instruction prompt for detoxifying toxic samples.

    Args:
        sample (dict): Dictionary containing 'toxic_sentence' and 'neutral_sentence' keys.
        mode (str): Mode of operation ('train' or 'test').

    Returns:
        dict: Dictionary containing the instruction prompt.
    """
    prompt = f"""
    Detoxify the following toxic sample.

    # {sample['toxic_sentence']}

    ### Detoxified:
    """
    if not isinstance(DETOX_INSTRUCTION, str):
        raise ValueError(f"prompt should be an str but got {type(DETOX_INSTRUCTION)}")

    if mode == "train":
        # TARGET = f"{Extract_KEY} \n {sample['neutral_sentence']}. "
        # final_prompt = (" ".join([INS_KEY, INSTRUCTION, CONTENT, TARGET, END_KEY]))
        # return {"query": final_prompt}
        return {"instruction": " ".join([DETOX_INS_KEY, prompt, sample['neutral_sentence'], DETOX_END_KEY])}

    elif mode == "test":
        # final_prompt = (" ".join([INS_KEY, INSTRUCTION, CONTENT, Extract_KEY]))
        # return {"query": final_prompt}
        return {"instruction": " ".join([DETOX_INS_KEY, prompt])}


def tsv_to_json(tsv_file_path, json_file_path):
    """
    Convert a TSV file to JSON format.

    Args:
        tsv_file_path (str): Path to the TSV file.
        json_file_path (str): Path to save the JSON file.
    """
    data = []
    with open(tsv_file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            data.append(row)
    print(len(data))
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


def remove_repeated_phrases(text):
    """
    Remove repeated phrases in a string.

    Args:
        text (str): Input text.

    Returns:
        str: Text with repeated phrases removed.
    """
    # Find repeated phrases using regex
    pattern = r'(\b.+?\b)(?:, \1\b)+'
    # Replace repeated phrases with a single occurrence
    cleaned_text = re.sub(pattern, r'\1', text)

    return cleaned_text


def generate_prompt(main_samples, mode):
    """
    Generate instruction prompts for detoxifying toxic samples.

    Args:
        main_samples (list): List of dictionaries containing toxic and neutral sentences.
        mode (str): Mode of operation ('train' or 'test').

    Returns:
        list: List of dictionaries containing instruction prompts.
    """
    instructions = []
    for data_sample in main_samples:
        # Generate the prompt using the separate function
        prompt = create_instruction(dict(data_sample), mode)  # use_ppo
        instructions.append(prompt)
    return instructions


def build_PPO_dataset(tokenizer, dataset_name):
    """
    Preprocess the dataset and split it into train and test parts.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for tokenizing the dataset.
        dataset_name (str): Name of the dataset to load.

    Returns:
        datasets.dataset_dict.DatasetDict: Preprocessed dataset containing train and test parts.
    """

    # load dataset (only "train" part will be enough for this lab).
    dataset = load_dataset(dataset_name, split="train")

    # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
    # dataset = dataset.filter(lambda x: len(x["en_toxic_comment"]) > input_min_text_length and len(x["en_toxic_comment"]) <= input_max_text_length, batched=False)

    # Prepare tokenizer. Setting device_map="auto" allows to switch between GPU and CPU automatically.
    def preprocess_function(sample):
        """
        Preprocess individual samples in the dataset.

        Args:
            sample (dict): A single sample from the dataset.

        Returns:
            dict: Preprocessed sample.
        """
        # Wrap each dialogue with the instruction.
        prompt = f"""
        Detoxify the following toxic sample.

        {sample['en_toxic_comment']}

        ### Detoxified: 
        """
        sample["input_ids"] = tokenizer.encode("### Instruction: " + prompt)

        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["neutral"] = sample["en_neutral_comment"]
        sample["toxic"] = sample["en_toxic_comment"]
        return sample

    # dataset = dataset[:100]
    # Tokenize each dialogue.
    dataset = dataset.map(preprocess_function, batched=False)
    dataset.set_format(type="torch")

    # Split the dataset into train and test parts.
    dataset_splits = dataset.train_test_split(test_size=0.10, shuffle=False, seed=42)

    return dataset_splits


def collator(data):
    """
    Custom collator function for organizing data.

    Args:
        data (list): List of data samples.

    Returns:
        dict: Organized data.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])


def create_DPO_instruction(sample, mode):
    """
    Generate an instruction prompt for DPO (Detoxify, Paraphrase, and Overwrite).

    Args:
        sample (dict): Dictionary containing toxic and neutral sentences.
        mode (str): Mode of operation ('train' or 'test').

    Returns:
        dict: Dictionary containing the instruction prompt.
    """
    start_key = "<|im_start|>"
    content_key = "### InputContent: "

    content = f"{content_key} \n {sample['toxic_sentence']}"

    chosen = sample['answer'] + "<|im_end|>\n"
    rejected = str(sample['distractor1']) + "<|im_end|>\n"
    if mode == "train":
        response = f"{DETOX_RES_KEY} \n {sample['answer']}"
        final_prompt = (" ".join(
            [start_key, DETOX_INSTRUCTION, DETOX_END_KEY, content, start_key, "assistant\n", response, DETOX_END_KEY]))
        return {"prompt": final_prompt, "chosen": chosen, "rejected": rejected}

    elif mode == "test":
        final_prompt = (" ".join([start_key, DETOX_INSTRUCTION, DETOX_END_KEY, content, start_key, "assistant\n", ]))
        return {"prompt": final_prompt, "chosen": chosen, "rejected": rejected}


if __name__ == '__main__':
    input_sample = [{"neutral_sentence": "you are shit", "toxic_sentence": "you are fucking shit"},
                    {"neutral_sentence": "He is bad", "toxic_sentence": "he is very awful"}]

    instruction = generate_prompt(input_sample, mode="train")
    print(instruction[0])

    print("\n")
    instruction = generate_prompt(input_sample, mode="train")
    print(instruction[0])
