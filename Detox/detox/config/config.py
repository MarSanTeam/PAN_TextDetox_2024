# -*- coding: utf-8 -*-
# ============================ Third Party libs ============================
import argparse
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import transformers


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--model_name", type=str, default="Mistral-7B-v0.1")

        self.parser.add_argument("--base_model_path", type=str, default="mnt/disk2/LanguageModels/Mistral-7B-v0.1")

        self.parser.add_argument("--pad_token", type=str, default="[PAD]")

        self.parser.add_argument("--eos_token", type=str, default="</s>")

        self.parser.add_argument("--bos_token", type=str, default="</s>")

        self.parser.add_argument("--unk_token", type=str, default="<unk>")

        self.parser.add_argument("--load_in_8bit", type=bool, default=False)

        self.parser.add_argument("--use_seq2seq", type=bool, default=False)

        self.parser.add_argument("--load_in_4bit", type=bool, default=True)

        self.parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True)

        self.parser.add_argument("--num_train_epochs", type=int, default=20)

        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

        self.parser.add_argument("--per_device_train_batch_size", type=int, default=32)
        self.parser.add_argument("--ppo_batch_size", type=int, default=4)
        self.parser.add_argument("--max_ppo_epochs", type=int, default=3)

        self.parser.add_argument("--logging_steps", type=int, default=1)

        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

        self.parser.add_argument("--max_length", type=int, default=90)

        self.parser.add_argument("--lora_r", type=int, default=32)

        self.parser.add_argument("--lora_alpha", type=int, default=64)

        self.parser.add_argument("--lora_dropout", type=float, default=0.05)

        self.parser.add_argument("--seed", type=int, default=5318008)

        self.parser.add_argument("--lora_rank", type=int, default=8)

        self.parser.add_argument("--optim", type=str, help="activates the paging for better memory management",
                                 default="paged_adamw_32bit", )

        self.parser.add_argument("--save_strategy", type=str, help="checkpoint save strategy", default="steps", )

        self.parser.add_argument("--evaluation_strategy", type=list, default=['no', 'steps', 'epoch'])

        self.parser.add_argument("--mode", type=list, default=['train', 'test'])

        self.parser.add_argument("--save_steps", type=int, help="checkpoint saves", default=50)

        self.parser.add_argument("--learning_rate", type=float, help="learning rate for AdamW optimizer",
                                 default=5e-5, )

        self.parser.add_argument("--max_grad_norm", type=float, help="maximum gradient norm (for gradient clipping)",
                                 default=0.3, )

        self.parser.add_argument("--max_steps", type=int, help="training will happen for 'max_steps' steps",
                                 default=500, )

        self.parser.add_argument("--input",

                                 default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/Evaluation/input.jsonl",
                                 help="Initial texts before style transfer", )

        self.parser.add_argument("-o", "--output", type=argparse.FileType("w", encoding="UTF-8"),
                                 default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/Evaluation/final_output.jsonl",
                                 help="Path where to write the evaluation results", )

        self.parser.add_argument( "--out_dir",
                                  default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_without",
                                 help="Path where to write the evaluation results", )
        self.parser.add_argument("--reference",
                                 default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/Evaluation/reference.jsonl",
                                 # type=argparse.FileType("rb"), required=False,
                                 help="Ground truth texts after style transfer", )

        self.parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable use of CUDA")

        self.parser.add_argument("--prediction",
                                 default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/Evaluation/prediction.jsonl",
                                 # type=argparse.FileType("rb"),
                                 help="Your model predictions")

        self.parser.add_argument("--language", required=False, type=str, default=None,
                                 help="Specify language. Should be one of "
                                      "['am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de']."
                                      " Without specification will load all stopwords.", choices=['ru', 'en'], )
        self.parser.add_argument("--warmup_ratio", type=float,
                                 help="steps for linear warmup from 0 " "to learning_rate", default=0.03, )
        self.parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
        self.parser.add_argument("--pattern", type=str, default=r"### Detoxified:\s*(.+?)\s*### End")
        self.parser.add_argument("--raw_pattern", type=str, default=r"#+\s*Detoxified:\s*(.+?)#+\s*End")

    def add_path(self) -> None:
        self.parser.add_argument("--raw_data_path", type=str, default=Path(__file__).parents[2].__str__() + "/data/Raw")
        self.parser.add_argument("--xlm_model_path", type=str, default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/toxicity_model_name/")
        self.parser.add_argument("--labse_model_path", type=str, default="/mnt/disk2/LanguageModels/labse/LaBSE/")
        self.parser.add_argument("--processed_data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed/")
        self.parser.add_argument("--data_path", type=str, default=Path(__file__).parents[2].__str__() + "/data/")
        self.parser.add_argument("--train_data_path", type=str, default="output.json", )
        self.parser.add_argument("--dev_data_path", type=str, default="output_dev.json")
        self.parser.add_argument("--data_mode", type=str, default=["EN", "RU"], )

    def get_config(self):
        self.add_path()
        return self.parser.parse_args()


@dataclass
class ModelArguments:
    SOLAR: str = "/mnt/disk2/LanguageModels/SOLAR-10.7B-v1.0"
    MISTRAL: str = "/mnt/disk2/LanguageModels/Mistral-7B-v0.1"
    MISTRALT5: str = "/mnt/disk2/LanguageModels/Mistral-T5-7B-v1"
    LAMA13: str = "/mnt/disk2/LanguageModels/llama-2-13b"
    INSTRUCT: str = "/mnt/disk2/LanguageModels/Mistral-7B-Instruct-v0.1"
    FALCON = "/mnt/disk2/LanguageModels/falcon-7b"
    FALCON_INS = "/mnt/disk2/LanguageModels/falcon-7b-instruct"
    ZEP = "/mnt/disk2/LanguageModels/zephyr-7b-beta"
    LLAMA2_13 = "/mnt/disk2/LanguageModels/llama-2-13b"
    LAMA3 = "/mnt/disk2/LanguageModels/LLama3"
    LAMA3_INS = "/mnt/disk2/LanguageModels/Llama3_8B_Instruct"
    GPT2 = "/mnt/disk2/LanguageModels/gpt2_large"
    runner_model_name_or_path: Optional[str] = field(default=ZEP)
    runner_tokenizer_name_or_path: Optional[str] = field(default=ZEP)
    model_name_or_path: Optional[str] = field(default=MISTRALT5)
    tokenizer_name_or_path: Optional[str] = field(default=MISTRALT5)
    ppo_fine_tuned_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/PPO/")
    fine_tuned_model_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_without")
    ru_fine_tuned_model_name_or_path_final: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_final_ru")
    en_fine_tuned_model_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_without")

    zh_fine_tuned_model_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_zh")
    ru_fine_tuned_model_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_ru")
    ru_fine_tuned_model_name_or_path2: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_ru2")
    without_fine_tuned_model_name_or_path: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_without")
    en_fine_tuned_model_name_or_path_final: Optional[str] = field(
        default="/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_simple_final")

    # mistral_simple, Lama3


@dataclass
class DataArguments:
    data_path: str = field(default="/mnt/disk2/maryam.najafi/Project_LLMFineTune/data/processed_data.json",
                           metadata={"help": "Path to the training data."}, )


@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


@dataclass
class TrainingArguments:
    output_dir: Optional[str] = field(default="")
    # per_device_train_batch_size:  Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    num_train_epochs: Optional[int] = field(default=5)
    save_total_limit: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=2e-4)
    warmup_ratio: Optional[float] = field(default=0.03)
    max_grad_norm: Optional[float] = field(default=0.3)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    evaluation_strategy: Optional[str] = field(default="steps")
    save_strategy: Optional[str] = field(default="steps")
    metric_for_best_model: Optional[str] = field(default="loss")
    bf16: Optional[bool] = field(default=False)
    tf32: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    group_by_length: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=True)
    logging_first_step: Optional[bool] = field(default=True)
    cache_dir: Optional[str] = field(default="")
    # optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=250)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
