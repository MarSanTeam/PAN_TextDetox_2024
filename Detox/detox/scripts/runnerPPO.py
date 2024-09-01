# -*- coding: utf-8 -*-
import logging

import pandas as pd
import torch
from peft import LoraConfig
from peft import PeftModel
from torch.optim import Adam
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
from transformers import set_seed, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

from detox import (BaseConfig, print_trainable_parameters)
from detox.config.config import ModelArguments
from detox.data_preparation.prepare_data import build_PPO_dataset, collator

tqdm.pandas()
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(filename='PPOtraining.log', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    set_seed(ARGS.seed)
    trl_model_class = AutoModelForCausalLMWithValueHead if not ARGS.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

    logger.info(f'Using device: {ARGS.device}')
    model_name = "/mnt/disk2/LanguageModels/Mistral-T5-7B-v1"
    output_dir = ModelArguments.ppo_fine_tuned_name_or_path
    peft_model_path = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_T5"

    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.bfloat16)

    if "lama" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        logger.info("Model is LAMA based LM!!!!!")
        # required for llama
        tokenizer.add_special_tokens(
            {"eos_token": "</s>", "bos_token": "</s>", "unk_token": "</s>", "pad_token": "[PAD]", })
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=ARGS.device)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f'Using device: {ARGS.device}')
    dataset = build_PPO_dataset(tokenizer=tokenizer, dataset_name="s-nlp/paradetox")
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", )
    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(model, peft_model_path, lora_config=lora_config, torch_dtype=torch.bfloat16,
                                           device_map=ARGS.device, is_trainable=True)
    logger.info(f'PEFT model parameters to be updated:\n{print_trainable_parameters(peft_model)}\n')
    logger.info(f'Using device: {ARGS.device}')

    """
    you just need to prepare the Proximal Policy Optimization (PPO) model passing the instruct-fine-tuned PEFT model
     to it. PPO will be used to optimize the RL policy against the reward model.
    """
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ARGS.learning_rate)

    ppo_model = trl_model_class.from_pretrained(peft_model, peft_config=lora_config, quantization_config=nf4_config,
                                                load_in_8bit=True, torch_dtype=torch.bfloat16, optimizer=optimizer,
                                                is_trainable=True)
    logger.info(f'Using device: {ARGS.device}')
    logger.info(
        f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_trainable_parameters(ppo_model)}\n')
    logger.info(ppo_model.v_head)

    logger.info(f'Using device: {ARGS.device}')
    """
    Now create a frozen copy of the PPO which will not be fine-tuned - a reference model. The reference model will
     represent the LLM before detoxification. None of the parameters of the reference model will be updated
      during PPO training. This is on purpose.
    """
    print("Using device:", ARGS.device)

    ref_model = create_reference_model(ppo_model, num_shared_layers=20)
    #
    logger.info(f'Reference model parameters to be updated:\n{print_trainable_parameters(ref_model)}\n')
    sentiment_pipe = pipeline("sentiment-analysis",
                              model="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/toxicity_model_name",
                              device=ARGS.device)

    """
    Set up the configuration parameters. Load the ppo_model and the tokenizer. You will also load a frozen
    version of the model ref_model. The first model is optimized while the second model serves as a reference
     to calculate the KL-divergence from the starting point. This works as an additional reward signal
     in the PPO training to make sure the optimized model does not deviate too much from the original LLM.
    """
    batch_size = 16
    max_ppo_epochs = 3
    mini_batch_size = 4
    print("dataset['train']", len(dataset["train"]))
    config = PPOConfig(model_name=model_name, learning_rate=ARGS.learning_rate, ppo_epochs=max_ppo_epochs,
                       mini_batch_size=mini_batch_size, batch_size=batch_size)

    ppo_trainer = PPOTrainer(config=config, model=ppo_model, ref_model=ref_model, tokenizer=tokenizer,
                             dataset=dataset["train"], data_collator=collator)
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    """
    The fine-tuning loop consists of the following main steps:

    - Get the query responses from the policy LLM (PEFT model).
    - Get sentiments for query/responses from hate speech RoBERTa model.
    - Optimize policy with PPO using the (query, response, reward) triplet.

    The operation is running if you see the following metrics appearing:

    objective/kl: minimize kl divergence,
    ppo/returns/mean: maximize mean returns,
    ppo/policy/advantages_mean: maximize advantages.
    """

    not_hate_index = 0
    output_min_length = 10
    output_max_length = 30

    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, }
    logger.info(f'Using device: {ARGS.device}')

    reward_kwargs = {"top_k": None,  # Return all scores.
                     "function_to_apply": "none",  # You want the raw logits without softmax.
                     "batch_size": batch_size}

    max_ppo_steps = 10

    filtered_responses = []

    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        # if step >= max_ppo_steps:
        #     break

        prompt_tensors = batch["input_ids"]
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()
            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        batch_responses = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        filtered_responses_batch = []
        print("batch query ", batch["query"])
        print("batch response ", batch_responses)
        logger.info(f'batch query, {batch["query"]}')
        logger.info(f'batch response, {batch_responses}')
        for query, response in zip(batch["query"], batch_responses):
            filtered_response = "### Detoxified: " + "".join(response.replace(query, ''))
            filtered_responses_batch.append(filtered_response)
        filtered_responses.extend(filtered_responses_batch)
        print("filtered_responses", filtered_responses)
        logger.info(f'filtered_responses, {filtered_responses}')

        query_response_pairs = [q + r for q, r in zip(batch["query"], filtered_responses_batch)]
        print("query_response_pairs", query_response_pairs)
        logger.info(f'query_response_pairs, {query_response_pairs}')

        # model = SentenceTransformer("/mnt/disk2/LanguageModels/labse/LaBSE")
        # embeddings_batch_responses = model.encode(batch_responses, convert_to_tensor=True)
        # embeddings_batch_target = model.encode(batch["en_neutral_comment"], convert_to_tensor=True)
        # similarities = util.pytorch_cos_sim(embeddings_batch_responses, embeddings_batch_target)
        # print("similarities", similarities)
        # similarities_reward_tensors = [torch.tensor(similarity) for similarity in similarities]
        # print("similarities_reward_tensors", similarities)
        rewards = sentiment_pipe(filtered_responses_batch, **reward_kwargs)
        # print("rewards", rewards)
        sentiment_reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]
        # print("sentiment_reward_tensors", sentiment_reward_tensors)
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, sentiment_reward_tensors)
        ppo_trainer.log_stats(stats, batch, sentiment_reward_tensors, columns_to_log=["query", "response"])

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))
        logger.info(f'objective/kl: {stats["objective/kl"]}')
        logger.info(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        logger.info(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        logger.info('-'.join('' for x in range(1000)))
        if epoch % 2 == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(output_dir)
    logger.info("Train Completed! Model saved ")
    print(" Train Completed!  Model saved ")
    # Save model every 100 epochs

    # trainer.save_model(output_dir)

    compare_results = {}

    df_batch = dataset["test"][0:batch_size]

    compare_results["query"] = df_batch["query"]
    prompt_tensors = df_batch["input_ids"]

    summary_tensors_ref = []
    summary_tensors = []

    # Get response from ppo and base model.
    for i in tqdm(range(batch_size)):
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        summary = peft_model.generate(input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to("cuda"),
                                      **generation_kwargs).squeeze()[-gen_len:]
        summary_tensors_ref.append(summary)

        summary = ppo_model.generate(input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to("cuda"),
                                     **generation_kwargs).squeeze()[-gen_len:]
        summary_tensors.append(summary)

    # Decode responses.
    compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
    compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

    # Sentiment analysis of query/response pairs before/after.
    texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
    rewards_before = sentiment_pipe(compare_results["response_before"], **reward_kwargs)
    compare_results["reward_before"] = [reward[not_hate_index]["score"] for reward in rewards_before]

    texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
    rewards_after = sentiment_pipe(compare_results["response_after"], **reward_kwargs)
    compare_results["reward_after"] = [reward[not_hate_index]["score"] for reward in rewards_after]
    pd.set_option('display.max_colwidth', 500)
    df_compare_results = pd.DataFrame(compare_results)
    df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
    df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(
        drop=True)
    df_compare_results_sorted.to_csv("df_compare_results_sorted.csv", index=False)
    print(df_compare_results_sorted)
    print(df_compare_results_sorted.head())
    logger.info(df_compare_results_sorted)
    logger.info(df_compare_results_sorted.head())
