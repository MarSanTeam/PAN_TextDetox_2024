# -*- coding: utf-8 -*-
import logging
import re
import time

import regex
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from detox.config.config import ModelArguments


class Predictor:
    """A class to make predictions using a fine-tuned language model."""

    def __init__(self, args, instructed_data, main_data, model_path, tuned_model):
        """
        Initializes the Predictor object.

        Args:
            args: Arguments.
            instructed_data (): List of dictionaries containing instruction data.
            main_data (list): List of dictionaries containing main data.
        """
        self.args = args
        self.instructed_data = instructed_data
        self.main_data = main_data
        self.logger = logging.getLogger(__name__)
        self.base_model = model_path
        self.tuned_model = tuned_model
        pass

    def setup_tokenizer(self):
        """
        Sets up the tokenizer for the provided model.

        Args:
            model_name_or_path (str): Model name or path.

        Returns:
            AutoTokenizer: The configured tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, add_bos_token=True, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load_finetuned_model(self):
        """
        Loads the fine-tuned model based on language.

        Returns:
            Tuple[PeftModel, AutoTokenizer]: The loaded model and tokenizer.
        """
        # Setup base model
        # lm_loader = LanguageModelLoader(ModelArguments.model_name_or_path, self.args.mode[1], self.args,
        #                                 self.instructed_data, self.instructed_data,
        #                                 ModelArguments.en_fine_tuned_model_name_or_path)
        # lm_loader.create_bnb_config()
        # base_model, _ = lm_loader.setup_model()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model,
                                                          quantization_config=bnb_config,
                                                          device_map="auto", torch_dtype=torch.bfloat16,
                                                          use_flash_attention_2=False, )
        # peft_config = lm_loader.create_peft_config()
        # model = lm_loader.peft_model_initializing()

        # Initialize model and tokenizer
        for sample, smpl in tqdm(zip(self.instructed_data, self.main_data)):
            lang = smpl.get("lang", "")

            if lang == "en":
                model_name_or_path = self.tuned_model
            # elif lang == "ru":
            #     model_name_or_path = ModelArguments.ru_fine_tuned_model_name_or_path_final
            else:
                print("Error")
                model_name_or_path = ModelArguments.fine_tuned_model_name_or_path

            print(f"\nthe given sample is {lang} and the loaded model is {model_name_or_path}\n")

            tokenizer = self.setup_tokenizer()
            peft_model = PeftModel.from_pretrained(base_model, model_name_or_path)

            # Additional operations if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                peft_model.is_parallelizable = True
                peft_model.model_parallel = True
            peft_model.eval()
            return peft_model, tokenizer

        # Default return if none of the conditions are met
        return None, None
    def load_org_finetuned_model(self):
        """
        Loads the fine-tuned model based on language.

        Returns:
            Tuple[PeftModel, AutoTokenizer]: The loaded model and tokenizer.
        """
        # Setup base model
        # lm_loader = LanguageModelLoader(ModelArguments.model_name_or_path, self.args.mode[1], self.args,
        #                                 self.instructed_data, self.instructed_data,
        #                                 ModelArguments.en_fine_tuned_model_name_or_path)
        # lm_loader.create_bnb_config()
        # base_model, _ = lm_loader.setup_model()

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model,
                                                          device_map="auto", torch_dtype=torch.bfloat16,
                                                          use_flash_attention_2=False, )
        # peft_config = lm_loader.create_peft_config()
        # model = lm_loader.peft_model_initializing()

        # Initialize model and tokenizer

        tokenizer = self.setup_tokenizer()


        return base_model, tokenizer

    def predict(self, peft_model, tokenizer):
        """
        Generates predictions using the provided model and tokenizer.

        Args:
            peft_model (PeftModel): The pre-trained language model.
            tokenizer (): The tokenizer for tokenizing input text.

        Returns:
            list: List of dictionaries containing predictions.
        """

        final_list = list()
        for sample, smpl in tqdm(zip(self.instructed_data, self.main_data)):
            config = {"max_new_tokens": len(tokenizer.encode(sample["instruction"])) + len(
                tokenizer.encode(smpl['toxic_sentence'])), "max_length": 70, "repetition_penalty": 1}
            with (torch.no_grad()):
                s_time =time.time()
                token_sample = tokenizer(sample["instruction"], return_tensors="pt", max_length=self.args.max_length,
                                         truncation=True).to("cuda")
                s_time1 =time.time()

                model_output = peft_model.generate(**token_sample, **config, pad_token_id=tokenizer.eos_token_id)


                results = [tokenizer.decode(gen, skip_special_tokens=True) for gen in model_output]

                print("\nThe generation middle time is:\n", time.time()-s_time1)
                pro_ins = PostProcess(self.args, results[0], self.instructed_data[0])
                processed_sample = pro_ins.extract_processed_sample()
                processed_sample = processed_sample.split('\n\n')[0].strip()
                print("smlp", smpl)
                print("The generated sample:", processed_sample)
                print("smpl['neutral_sentence']", smpl['neutral_sentence'])
                print("\nThe generation time is:\n", time.time()-s_time)
                final_list.append({"input": smpl["toxic_sentence"], "prediction": processed_sample,
                                   "reference": smpl['neutral_sentence']})
        print(final_list)

        return final_list

    def predict_batch(self, peft_model, tokenizer):
        """
        Generates predictions in batches using the provided model and tokenizer.

        Args:
            peft_model (PeftModel): The pre-trained language model.
            tokenizer (): The tokenizer for tokenizing input text.

        Returns:
            list: List of dictionaries containing predictions.
        """
        peft_model.eval()
        final_list = list()
        for sample, smpl in tqdm(zip(self.instructed_data, self.main_data)):
            config = {"max_new_tokens": len(tokenizer.encode(sample["instruction"])) + len(
                tokenizer.encode(smpl['toxic_sentence'])), "repetition_penalty": 1}
            input_instructions = [sample["instruction"]]
            token_samples = tokenizer(input_instructions, padding=True, return_tensors="pt", truncation=True).to(
                "cuda:0")

            # Generate outputs in batches
            for i in range(0, len(token_samples["input_ids"]), self.args.batch_size):
                batch_inputs = {k: v[i:i + self.args.batch_size] for k, v in token_samples.items()}
                with torch.no_grad():
                    model_output = peft_model.generate(**batch_inputs, **config, pad_token_id=tokenizer.eos_token_id)
                    results = [tokenizer.decode(gen, skip_special_tokens=True) for gen in model_output]
                    pro_ins = PostProcess(self.args, results[0], self.instructed_data[0])
                    processed_sample = pro_ins.extract_processed_sample()
                    final_list.append({"input": smpl["toxic_sentence"], "prediction": processed_sample,
                                       "reference": smpl['neutral_sentence']})

        print(final_list)
        return final_list


class PostProcess:
    """A class to post-process generated samples."""

    def __init__(self, args, generated_sample, text_to_ignore):
        """
        Initializes the PostProcess object.

        Args:
            generated_sample (str): The generated sample text.
            text_to_ignore (str): The text to be ignored during processing.
        """
        self.generated_sample = generated_sample
        self.text_to_ignore = text_to_ignore
        self.args = args

    def ignore_initial_text(self):
        """
        Removes initial text to be ignored from the generated sample.

        Returns:
            str: The filtered output with ignored text removed.
        """
        filtered_output = self.generated_sample.replace(self.text_to_ignore, '')
        return "### Detoxified:" + filtered_output

    @staticmethod
    def raw_extract(to_be_filtered):
        """
        Performs raw extraction from the text to be filtered.

        Args:
            to_be_filtered (str): The text to be filtered.

        Returns:
            str: The extracted text.
        """
        detoxified_sentence = regex.search(r'### Detoxified:\s*(.*?)(?=\n)', to_be_filtered).group(1)

        return detoxified_sentence  # .strip()

    def final_process(self):
        """
        Executes the final processing steps.

        Returns:
            str: The processed sample.
        """
        filtered_text = self.ignore_initial_text()
        # extracted = self.raw_extract(filtered_text)
        extracted = self.extract_processed_sample()
        return extracted

    def extract_processed_sample(self):
        """
          Extracts the processed sample from the generated text.

          Args:
          Returns:
              str: The processed sample.
          """
        match = re.search(self.args.pattern, self.generated_sample, re.DOTALL)

        if match:
            processed_sample = match.group(1).strip()
        else:
            matched = re.search(self.args.raw_pattern, self.generated_sample, flags=re.DOTALL)
            processed_sample = matched.group(1).strip() if matched else self.generated_sample

        return processed_sample.replace(".", "").replace("### End", "").replace("### Detoxified:", "")
