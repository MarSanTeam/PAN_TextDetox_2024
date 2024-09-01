# -*- coding: utf-8 -*-

import bitsandbytes as bnb


def get_max_length(model):
    """
    Get the maximum length of tokens supported by the model.

    Args:
        model: Model object.

    Returns:
        int: Maximum length of tokens.

    """
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def print_trainable_parameters(model):
    """
    Print information about the model's trainable parameters.

    Args:
        model: Model object.

    Returns:
        str: Information about trainable parameters.

    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return (f"trainable params: {trainable_params} ||"
            f" all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def find_all_linear_names(model):
    """
    Find names of all linear modules in the model.

    Args:
        model: Model object.

    Returns:
        list: List of linear module names.

    """
    cls = (bnb.nn.Linear4bit)  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        #    if 'lm_head' in lora_module_names:  # needed for 16-bit
        #        lora_module_names.remove('lm_head')
        return list(lora_module_names)


def calculate_reward(predicted_sample, toxicity_tokenizer, toxicity_model):
    """
    Calculate the reward based on predicted sample's toxicity.

    Args:
        predicted_sample (str): Predicted text sample.
        toxicity_tokenizer: Tokenizer for toxicity model.
        toxicity_model: Toxicity classification model.

    Returns:
        list: List of toxicity rewards.

    """
    toxicity_input_ids = toxicity_tokenizer(predicted_sample, return_tensors="pt").input_ids.to("cuda")

    logits = toxicity_model(input_ids=toxicity_input_ids).logits
    print(f'logits [not hate, hate]: {logits.tolist()[0]}')

    # Print the probabilities for [not hate, hate]
    probabilities = logits.softmax(dim=-1).tolist()[0]
    print(f'probabilities [not hate, hate]: {probabilities}')

    # get the logits for "not hate" - this is the reward!
    not_hate_index = 0
    toxicity_reward = (logits[:, not_hate_index]).tolist()
    print(f'reward (high): {toxicity_reward}')
    return toxicity_reward
