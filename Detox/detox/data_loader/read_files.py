# -*- coding: utf-8 -*-
import json

import numpy as np

from detox.data_loader.helper import _make_r_io_base


def jload(file_path, mode="r"):
    """
    Load a .json file into a dictionary.

    Args:
        file_path (str or IO): The path or file-like object of the .json file to load.
        mode (str): The mode to open the file, defaults to "r" (read).

    Returns:
        dict: The loaded JSON data as a dictionary.

    Examples:
        jdict = jload("example.json")
    """
    f = _make_r_io_base(file_path, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def load_jsonl(file_path, mode="r"):
    """
    Load data from a .jsonl file into a list of dictionaries.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list containing dictionaries loaded from the .jsonl file.

    Examples:
        data_list = load_jsonl("example.jsonl")
        :param file_path:  The path to the .jsonl file.
        :param mode: boolean flag
    """
    data_list = list()
    with open(file_path, mode, encoding="utf8") as file:
        for line in file:
            json_object = json.loads(line.strip())
            data_list.append(json_object)
    return data_list


def load_json(file_path):
    """
    Load data from a .json file into a dictionary.

    Args:
        file_path (str): The path to the .json file.

    Returns:
        dict: The loaded JSON data as a dictionary.

    Examples:
        data = load_json("example.json")
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return data


def write_json(out_dir, input_data, mode="w"):
    """
    Write a dictionary to a .json file.

    Args:
        out_dir (str): The path to the output .json file.
        input_data (dict): The dictionary to be written to the .json file.

    Returns:
        None

    Examples:
        write_json("output.json", {"key": "value"})
        :param input_data: he path to the output .json file.
        :param out_dir: The dictionary to be written to the .json file.
        :param mode: the mode of data loader
    """
    with open(out_dir, mode, ) as test_file:
        json.dump(input_data, test_file, ensure_ascii=False)


def load_npy(file_path):
    """
    Load data from a .numpy file into a dictionary.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        dict: The loaded JSON data as a dictionary.

    Examples:
        data = load_npy("example.npy")
    """
    return np.load(file_path, allow_pickle=True)


def dump_txt(file_path: str, input_data):
    """
    Load data from a .txt file into a dictionary.

    Args:
        file_path (str): The path to the .txt file.
        input_data

    Returns:
        dict: The loaded txt data as a dictionary.

    Examples:
        data = dump_txt("example.txt")
    """
    with open(file_path, 'w') as file:
        # Iterate through the list and write each element on a new line
        for item in input_data:
            file.write(f"{item}\n")
