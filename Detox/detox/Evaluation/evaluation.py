#!/usr/bin/env python3

__credits__ = ["David Dale", "Daniil Moskovskiy", "Dmitry Ustalov", "Elisei Stakovskii"]

import argparse
import json
import os
from functools import partial
from typing import Optional, Type, Tuple, Dict, Callable, List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sacrebleu import CHRF
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, )

from detox import BaseConfig
from detox.config.config import ModelArguments


class Evaluation:
    def __init__(self, args: argparse.Namespace, samples_df: pd.DataFrame, out_dir):
        self.args = args
        self.samples_df = samples_df
        self.out_dir= out_dir

    def prepare_target_label(self, model: AutoModelForSequenceClassification, target_label: Union[int, str]) -> int:
        """
        Prepare the target label to ensure it is valid for the given model.

        Args:
            model (AutoModelForSequenceClassification): Text classification model.
            target_label (Union[int, str]): The target label to prepare.

        Returns:
            int: The prepared target label.

        Raises:
            ValueError: If the target_label is not found in model labels or ids.
        """
        if target_label in model.config.id2label:
            pass
        elif target_label in model.config.label2id:
            target_label = model.config.label2id.get(target_label)
        elif (isinstance(target_label, str) and target_label.isnumeric() and int(
                target_label) in model.config.id2label):
            target_label = int(target_label)
        else:
            raise ValueError(f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.')
        assert isinstance(target_label, int)
        return target_label

    def classify_texts(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: List[str],
                       target_label: Union[int, str], second_texts: Optional[List[str]] = None, batch_size: int = 32,
                       raw_logits: bool = False, desc: Optional[str] = "Calculating STA scores", ) -> npt.NDArray[
        np.float64]:
        """
        Classify a list of texts using the given model and tokenizer.

        Args:
            model (AutoModelForSequenceClassification): Text classification model.
            tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
            texts (List[str]): List of texts to classify.
            target_label (Union[int, str]): The target label for classification.
            second_texts (Optional[List[str]]): List of secondary texts (not needed by default).
            batch_size (int): Batch size for inference.
            raw_logits (bool): Whether to return raw logits instead of probs.
            desc (Optional[str]): Description for tqdm progress bar.

        Returns:
            npt.NDArray[np.float64]: Array of classification scores for the texts.
        """

        target_label = self.prepare_target_label(model, target_label)

        res = []

        for i in trange(0, len(texts), batch_size, desc=desc):
            inputs = [texts[i: i + batch_size]]

            if second_texts is not None:
                inputs.append(second_texts[i: i + batch_size])
            inputs = tokenizer(*inputs, return_tensors="pt", padding=True, truncation=True, max_length=512, ).to(
                model.device)

            with torch.no_grad():
                try:
                    logits = model(**inputs).logits
                    if raw_logits:
                        preds = logits[:, target_label]
                    elif logits.shape[-1] > 1:
                        preds = torch.softmax(logits, -1)[:, target_label]
                    else:
                        preds = torch.sigmoid(logits)[:, 0]
                    preds = preds.cpu().numpy()
                except:
                    print(i, i + batch_size)
                    preds = [0] * len(inputs)
            res.append(preds)
        return np.concatenate(res)

    def evaluate_sta(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: List[str],
                     target_label: int = 1,  # 1 is polite, 0 is toxic
                     batch_size: int = 32, ) -> npt.NDArray[np.float64]:
        """
        Evaluate the STA of a list of texts using the given model and tokenizer.

        Args:
            model (AutoModelForSequenceClassification): Text classification model.
            tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
            texts (List[str]): List of texts to evaluate.
            target_label (int): The target label for style evaluation.
            batch_size (int): Batch size for inference.

        Returns:
            npt.NDArray[np.float64]: Array of STA scores for the texts.
        """
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(model, tokenizer, texts, target_label, batch_size=batch_size, desc="Style")

        return scores

    def evaluate_sim(self, model: SentenceTransformer, original_texts: List[str], rewritten_texts: List[str],
                     batch_size: int = 32, efficient_version: bool = False, ) -> npt.NDArray[np.float64]:
        """
        Evaluate the semantic similarity between original and rewritten texts.
        Note that the subtraction is done due to the implementation of the `cosine` metric in `scipy`.
        For more details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

        Args:
            model (SentenceTransformer): The sentence transformer model.
            original_texts (List[str]): List of original texts.
            rewritten_texts (List[str]): List of rewritten texts.
            batch_size (int): Batch size for inference.
            efficient_version (bool): To use efficient calculation method.

        Returns:
            npt.NDArray[np.float64]: Array of semantic similarity scores between \
                  original and rewritten texts.
        """
        similarities = []

        batch_size = min(batch_size, len(original_texts))
        for i in trange(0, len(original_texts), batch_size, desc="Calculating SIM scores"):
            original_batch = original_texts[i: i + batch_size]
            rewritten_batch = rewritten_texts[i: i + batch_size]

            embeddings = model.encode(original_batch + rewritten_batch)
            original_embeddings = embeddings[: len(original_batch)]
            rewritten_embeddings = embeddings[len(original_batch):]

            if efficient_version:
                similarity_matrix = np.dot(original_embeddings, rewritten_embeddings.T)
                original_norms = np.linalg.norm(original_embeddings, axis=1)
                rewritten_norms = np.linalg.norm(rewritten_embeddings, axis=1)
                similarities.extend(1 - similarity_matrix / (np.outer(original_norms, rewritten_norms) + 1e-9))
            else:
                t = [1 - cosine(original_embedding, rewritten_embedding) for original_embedding, rewritten_embedding in
                     zip(original_embeddings, rewritten_embeddings)]
                similarities.extend(t)
        return similarities

    def evaluate_style_transfer(self, original_texts: List[str], rewritten_texts: List[str],
                                style_model: AutoModelForSequenceClassification, style_tokenizer: AutoTokenizer,
                                meaning_model: AutoModelForSequenceClassification,
                                references: Optional[List[str]] = None, style_target_label: int = 1,
                                batch_size: int = 32, ) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Wrapper for calculating sub-metrics and joint metric.

        Args:
            original_texts (List[str]): List of original texts.
            rewritten_texts (List[str]): List of rewritten texts.
            style_model (AutoModelForSequenceClassification): The style classification model.
            style_tokenizer (AutoTokenizer): The tokenizer corresponding to the style model.
            meaning_model (AutoModelForSequenceClassification): The meaning classification model.
            references (Optional[List[str]]): List of reference texts (if available).
            style_target_label (int): The target label for style classification.
            batch_size (int): Batch size for inference.

        Returns:
            Dict[str, npt.NDArray[np.float64]]: Dictionary containing evaluation metrics.
        """
        accuracy = self.evaluate_sta(style_model, style_tokenizer, rewritten_texts, target_label=style_target_label,
                                     batch_size=batch_size, )

        similarity = self.evaluate_sim(model=meaning_model, original_texts=original_texts,
                                       rewritten_texts=rewritten_texts, batch_size=batch_size, )

        result = {"STA": accuracy, "SIM": similarity, }

        if references is not None:
            chrf = CHRF()

            result["CHRF"] = np.array(
                [chrf.sentence_score(hypothesis=rewritten, references=[reference]).score / 100 for rewritten, reference
                 in zip(rewritten_texts, references)], dtype=np.float64, )

            result["J"] = result["STA"] * result["SIM"] * result["CHRF"]

        return result

    def load_model(self, model_name: Optional[str] = None, model: Optional[AutoModelForSequenceClassification] = None,
                   tokenizer: Optional[AutoTokenizer] = None,
                   model_class: Type[AutoModelForSequenceClassification] = AutoModelForSequenceClassification,
                   use_cuda: bool = True, ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load a pre-trained model and tokenizer from Hugging Face Hub.

        Args:
            model_name (Optional[str]): The name of the model to load.
            model (Optional[AutoModelForSequenceClassification]): A pre-loaded model instance.
            tokenizer (Optional[AutoTokenizer]): A pre-loaded tokenizer instance.
            model_class (Type[AutoModelForSequenceClassification]): The class of the model to load.
            use_cuda (bool): Whether to use CUDA for GPU acceleration.

        Returns:
            Tuple[AutoModelForSequenceClassification, AutoTokenizer]: The loaded model and tokenizer.
        """
        if model_name == self.args.labse_model_path:
            model = SentenceTransformer(self.args.labse_model_path)
            return model
        if model is None:
            if model_name is None:
                raise ValueError("Either model or model_name should be provided")
            model = model_class.from_pretrained(model_name, device_map='cuda')

            # if torch.cuda.is_available() and use_cuda:  #     model.cuda()
        if tokenizer is None:
            if model_name is None:
                raise ValueError("Either tokenizer or model_name should be provided")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def format_prototext(self, measure: str, value: str) -> str:
        """
        Format evaluation metrics into prototext format.

        Args:
            measure (str): The name of the evaluation measure.
            value (str): The value of the evaluation measure.

        Returns:
            str: The formatted prototext string.
        """
        return f'measure{{\n  key: "{measure}"\n  value: "{value}"\n}}\n'

    def run_evaluation(self, evaluator: Callable[..., Dict[str, npt.NDArray[np.float64]]], ) -> Dict[
        str, npt.NDArray[np.float64]]:
        """
        Run evaluation on input data using the specified evaluator.

        Args:
            evaluator (Callable[..., Dict[str, npt.NDArray[np.float64]]]): The evaluation function.

        Returns:
            Dict[str, npt.NDArray[np.float64]]: Dictionary containing evaluation results.
        """

        assert not self.samples_df.isna().values.any(), "Datasets contain missing entries"
        # if self.samples_df.reference:
        #     assert (len(self.sa
        #     mples_df["input"]) == len(self.samples_df["prediction"]) == len(
        #         self.samples_df["reference"])), (
        #         f"Dataset lengths {len(self.samples_df['input'])} & {len(self.samples_df['prediction'])}"
        #         f" !=  {len(self.samples_df['reference'])}")
        assert (len(self.samples_df["input"]) == len(self.samples_df["prediction"])), (
            f"Dataset lengths {len(self.samples_df['input'])}!= {len(self.samples_df['prediction'])}")

        result = evaluator(original_texts=self.samples_df["input"].tolist(),
                           rewritten_texts=self.samples_df["prediction"].tolist(),
                           references=self.samples_df["reference"].tolist() or None, )

        aggregated = {measure: np.mean(values).item() for measure, values in result.items()}

        print(aggregated)
        with open(self.out_dir, 'w') as json_file:
            json.dump(aggregated, json_file)
        for measure, value in aggregated.items():

            self.args.output.write(self.format_prototext(measure, str(value)))
        return result

    def main(self) -> None:
        style_model, style_tokenizer = self.load_model(self.args.xlm_model_path)  # , use_cuda=not ARGS.no_cuda)
        meaning_model = self.load_model(self.args.labse_model_path)  # , use_cuda=not ARGS.no_cuda)

        self.run_evaluation(
            evaluator=partial(self.evaluate_style_transfer, style_model=style_model, style_tokenizer=style_tokenizer,
                              meaning_model=meaning_model, style_target_label=0, ), )


if __name__ == "__main__":
    import torch

    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    SAMPLES_DF = pd.read_csv("/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/detox/Evaluation/joint_df.csv")
    evaluation = Evaluation(ARGS, SAMPLES_DF)
    model, tokenizer = evaluation.load_model("/mnt/disk2/LanguageModels/deto2")
    model.eval()
    print(model.config.id2label)  # {+: 'LABEL_0', -: 'LABEL_1'}

    evaluation.main()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    texts = [
    "Stupid bitch is ugly inside as well as out",
    "Arnt all Trump Supporters White Supremacist.",
    "joe rogan is a retard cause if a man is a feminist he must looowe pussy fuck working for a cause",
    "Not for your disgusting little beta hands‍ Wipe away those tears, or just use them to jerk off to this video, you pathetic cuck.",
    "Holy fuck i would love those tits in my face as you rode my hard cock",
    "I will fucking destroy ur bloodline u fucking bitch",
    "Spitting Girls, Cam Şov, Desi porn, Swedish, Liseli porno, British Sex, pissing girl, Chat sitesi, URL",
    "You'd hate them, too, if you didn't want to ram your cock in-and-out of them.",
    "This happens every fucking time qith you whites.",
    "Fucking xians can suck my motherfucking dick.",
    "Don't project your grandpas faggotry on mine"]
    # text = ["I like you. I love you"]
    # scores = evaluation.evaluate_sta(model, tokenizer, non_toxic_text)
    # print(scores)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    #
    for text in texts:
        xlmr_model = AutoModelForSequenceClassification.from_pretrained("/mnt/disk2/LanguageModels/deto2")
        xlmr_tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/LanguageModels/deto2")

        # non_toxic_text = "love you"
        toxicity_input_ids = xlmr_tokenizer(text, return_tensors="pt").input_ids

        logits = xlmr_model(input_ids=toxicity_input_ids).logits
        print(f'logits [good, bad]: {logits.tolist()[0]}')

        # Print the probabilities for [good, bad]
        probabilities = logits.softmax(dim=-1).tolist()[0]
        print(f'probabilities [good, bad]: {probabilities}')

        # get the logits for "good" - this is the reward!
        not_hate_index = 0
        nothate_reward = (logits[:, not_hate_index]).tolist()
        print(f'reward (high): {nothate_reward}')

        predicted_class = torch.argmax(logits, dim=1).item()
        print("Predicted Class:", predicted_class)
