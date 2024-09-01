# -*- coding: utf-8 -*-
from typing import Tuple

import bitsandbytes as bnb
import torch
import transformers
from datasets.utils.logging import set_verbosity_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer)
from transformers import (LlamaTokenizer)
from trl import SFTTrainer, AutoModelForCausalLMWithValueHead
from trl import create_reference_model

from detox.config.config import ModelArguments
from detox.models.helper import print_trainable_parameters


class LanguageModelLoader:
    def __init__(self, model_name, mode, arg, train_data, eval_data, out_dir):
        self.model_args = None
        self.model_name = model_name
        self.arg = arg
        self.mode = mode
        self.train_data = train_data
        self.eval_data = eval_data
        # self.accelerator = accelerator
        self.model, self.tokenizer = self.setup_model()
        self.out_dir = out_dir
        self.num_evaluate_steps = int(
            len(self.train_data) / (self.arg.per_device_train_batch_size * self.arg.gradient_accumulation_steps))
        self.model_args = transformers.TrainingArguments(per_device_train_batch_size=6, gradient_accumulation_steps=4,
                                                         learning_rate=5e-5, lr_scheduler_type="cosine", max_steps=200,
                                                         save_strategy="no", logging_steps=self.arg.logging_steps,
                                                         output_dir=ModelArguments.fine_tuned_model_name_or_path,
                                                         num_train_epochs=12, optim="paged_adamw_32bit",
                                                         warmup_steps=100, bf16=False, report_to="tensorboard",
                                                         load_best_model_at_end=True, save_total_limit=2)

    @staticmethod
    def create_bnb_config():
        """
        Creates and returns a BitsAndBytesConfig for model quantization.

        Returns:
            BitsAndBytesConfig: Configuration for model quantization with the following parameters:
                - load_in_4bit: Whether to load the model in 4-bit mode.
                - bnb_4bit_use_double_quant: Whether to use double quantization for 4-bit mode.
                - bnb_4bit_quant_type: The type of 4-bit quantization, e.g., "nf4".
                - bnb_4bit_compute_dtype: The data type used for 4-bit computation, e.g., torch.bfloat16.

        Examples:
            bnb_config = create_bnb_config()
        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16, )
        return bnb_config

    def create_peft_config(self):
        """
        Create Parameter-Efficient Fine-Tuning (PEFT) configuration for the given model.

        Args:

        Returns:
            LoraConfig: Configuration for parameter-efficient fine-tuning using Lora with the following parameters:
                - r: Dimension of the updated matrices.
                - lora_alpha: Parameter for scaling.
                - target_modules: Names of the target modules to apply Lora to.
                - lora_dropout: Dropout probability for layers.
                - bias: Bias type, e.g., "none".
                - task_type: Task type, e.g., "CAUSAL_LM".

        Examples:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            lm_loader = LanguageModelLoader("gpt2")
            peft_config = lm_loader.create_peft_config(model)
        """
        config = LoraConfig(r=self.arg.lora_r,  # dimension of the updated matrices
                            lora_alpha=self.arg.lora_alpha,  # parameter for scaling
                            target_modules=self.find_all_linear_names(),  # target_modules=find_all_linear_names(model)
                            lora_dropout=self.arg.lora_dropout,  # dropout probability for layers
                            bias="none", task_type="CAUSAL_LM", )
        return config

    def peft_model_initializing(self):
        """
        Initialize a language model for Parameter-Efficient Fine-Tuning (PEFT).

        Args:

        Returns:
            Model: Parameter-Efficient Fine-Tuning initialized model.

        Examples:
            arg_namespace = Namespace(pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
            lm_loader = LanguageModelLoader("gpt2")
            peft_model = lm_loader.peft_model_initializing(arg_namespace)
        """
        peft_config = self.create_peft_config()
        model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
        peft_model = get_peft_model(model, peft_config)
        return peft_model

    def find_all_linear_names(self):
        """
        Find all linear module names in the given language model.

        Args:

        Returns:
            list: List of linear module names.

        Examples:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            lm_loader = LanguageModelLoader("gpt2")
            linear_module_names = lm_loader.find_all_linear_names(model)
        """
        cls = (bnb.nn.Linear4bit)  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        #    if 'lm_head' in lora_module_names:  # needed for 16-bit
        #        lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def setup_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads a pre-trained language model and tokenizer with specified configurations.

        Returns:
            Tuple: A tuple containing the loaded language model (AutoModelForCausalLM) and tokenizer (AutoTokenizer).
            Tuple[PreTrainedModel, PreTrainedTokenizer]: model and tokenizer
        Raises:
            SomeException: Description of the exception, if any.

        Examples:
            arg = Namespace(pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
            model_name = "ex: gpt2"
            bnb_config = QuantizationConfig(...)
            model, tokenizer = setup_model(arg, model_name, bnb_config)
        """

        model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.create_bnb_config(),
                                                     device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=True,
                                                     use_flash_attention_2=False, )
        model.config.use_cache = False
        if torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        if "lama" in self.model_name:
            tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            print("Model is LAMA based LM!!!!!")
            tokenizer.add_special_tokens(
                {"eos_token": "</s>", "bos_token": "</s>", "unk_token": "</s>", "pad_token": "[PAD]", })
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map=self.arg.device)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer

    def forward(self):
        """

        Returns:

        """
        model, tokenizer = self.setup_model()

        # Create PEFT configuration
        peft_config = self.create_peft_config()
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False
        # Check GPU compatibility with bfloat16
        bnb_4bit_compute_dtype = "float16"
        use_4bit = True
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        model = self.peft_model_initializing()
        print_trainable_parameters(model)
        trainer = SFTTrainer(model=model, train_dataset=self.train_data, eval_dataset=self.eval_data,
                             peft_config=peft_config, max_seq_length=self.arg.max_length, tokenizer=tokenizer,
                             args=self.model_args, dataset_text_field="instruction", )
        model.config.use_cache = False

        # Set verbosity to info
        set_verbosity_info()

        # Train the model
        trainer.train()

        # Save the last checkpoint of the model
        print("Saving last checkpoint of the model...")
        trainer.save_model(self.out_dir)

    def run_and_merge(self, model_name):
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(self.model, low_cpu_mem_usage=True, return_dict=True,
                                                          torch_dtype=torch.float16, device_map="auto", )
        model = PeftModel.from_pretrained(base_model, ModelArguments.fine_tuned_model_name_or_path)
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.push_to_hub(ModelArguments.fine_tuned_model_name_or_path, use_temp_dir=False)
        tokenizer.push_to_hub(ModelArguments.fine_tuned_model_name_or_path, use_temp_dir=False)  #


class LanguageModelLoaderPPO:
    def __init__(self, model_name, arg, data_len):
        self.model_args = None
        self.model_name = model_name
        self.arg = arg
        self.data_len = data_len
        self.model, self.tokenizer = self.setup_model()

    def create_peft_config(self):
        """
        Create Parameter-Efficient Fine-Tuning (PEFT) configuration for the given model.

        Args:

        Returns:
            LoraConfig: Configuration for parameter-efficient fine-tuning using Lora with the following parameters:
                - r: Dimension of the updated matrices.
                - lora_alpha: Parameter for scaling.
                - target_modules: Names of the target modules to apply Lora to.
                - lora_dropout: Dropout probability for layers.
                - bias: Bias type, e.g., "none".
                - task_type: Task type, e.g., "CAUSAL_LM".

        Examples:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            lm_loader = LanguageModelLoader("gpt2")
            peft_config = lm_loader.create_peft_config(model)
        """
        config = LoraConfig(r=32, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none",
                            task_type="CAUSAL_LM")

        return config

    def setup_model(self):
        """
        Loads a pre-trained language model and tokenizer with specified configurations.

        Returns:
            Tuple: A tuple containing the loaded language model (AutoModelForCausalLM) and tokenizer (AutoTokenizer).

        Raises:
            SomeException: Description of the exception, if any.

        Examples:
            arg = Namespace(pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
            model_name = "gpt2"
            bnb_config = QuantizationConfig(...)
            model, tokenizer = setup_model(arg, model_name, bnb_config)
        """
        self.model_args = transformers.TrainingArguments(per_device_train_batch_size=6, gradient_accumulation_steps=4,
                                                         learning_rate=5e-5, lr_scheduler_type="cosine", max_steps=200,
                                                         save_strategy="no", logging_steps=self.arg.logging_steps,
                                                         output_dir=ModelArguments.fine_tuned_model_name_or_path,
                                                         num_train_epochs=12, optim="paged_adamw_32bit",
                                                         warmup_steps=100, bf16=False, report_to="tensorboard",
                                                         logging_dir=ModelArguments.fine_tuned_model_name_or_path,
                                                         load_best_model_at_end=True, save_total_limit=2)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.arg.device,
                                                     torch_dtype=torch.bfloat16, load_in_8bit=True,
                                                     use_flash_attention_2=False, )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        return model, tokenizer

    def forward(self):
        model, tokenizer = self.setup_model()

        # Create PEFT configuration
        lora_config = self.create_peft_config()
        peft_model = PeftModel.from_pretrained(model, ModelArguments.fine_tuned_model_name_or_path,
                                               lora_config=lora_config, torch_dtype=torch.bfloat16,
                                               device_map=self.arg.device, is_trainable=True)
        print(f'PEFT model parameters to be updated:\n{print_trainable_parameters(peft_model)}\n')
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model, torch_dtype=torch.bfloat16,
                                                                      is_trainable=True)
        print(
            f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_trainable_parameters(ppo_model)}\n')
        print(ppo_model.v_head)
        ref_model = create_reference_model(ppo_model)
        print(f'Reference model parameters to be updated:\n{print_trainable_parameters(ref_model)}\n')
        return model, tokenizer, peft_model, ppo_model, ref_model
