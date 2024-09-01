import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import set_seed

from detox import generate_prompt
from detox.Evaluation import Evaluation
from detox.config import BaseConfig
from detox.inference.inference import Predictor

tqdm.pandas()
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
handler = RotatingFileHandler('training.log', maxBytes=1000000, backupCount=1)
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    set_seed(ARGS.seed)
    # --------------------------------------------- Load Data -----------------------------------------------
    dev_data = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_dev.tsv"
    test_data = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_test.tsv"
    df = pd.read_csv(dev_data, sep="\t")
    DEV_DATA_EN = df[df['lang'] == 'en']
    DEV_DATA_EN = DEV_DATA_EN.to_dict(orient='records')
    print("\n Dev Data type is: {}".format(type(DEV_DATA_EN)))
    print("\n Dev Data sample: {}".format(DEV_DATA_EN[1]))
    logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA_EN)))
    logging.warning("\n Dev Data sample: {}".format(DEV_DATA_EN[1]))
    print("***" * 10)
    # --------------------------------------------- Create Prompt -----------------------------------------------
    instructed_DEV_DATA = generate_prompt(DEV_DATA_EN, mode="test")
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[0])))
    instructed_DEV_DATA = Dataset.from_list(instructed_DEV_DATA)
    logging.warning("\n Dev prompted Data length is: {}".format(len(instructed_DEV_DATA)))
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[0])))
    logging.warning(f" {len(DEV_DATA_EN)} is dev data len...")
    print("***" * 10)

    # --------------------------------------------- Generate Prediction -----------------------------------------------
    base_model_path = "/mnt/disk2/LanguageModels/Mistral-T5-7B-v1"
    tuned_model_path = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/mistral_T5"
    print("based_model_name_or_path", base_model_path)
    print("tuned_model_name_or_path", tuned_model_path)

    PRED_INS = Predictor(ARGS, instructed_DEV_DATA, DEV_DATA_EN, base_model_path, tuned_model_path)
    PEFT_MODEL, TOKENIZER = PRED_INS.load_org_finetuned_model()
    FINAL_LIST = PRED_INS.predict(PEFT_MODEL, TOKENIZER)
    print(FINAL_LIST)
    # --------------------------------------------- Evaluate Prediction -----------------------------------------------
    FINAL_DF = pd.DataFrame(FINAL_LIST)
    FINAL_DF.to_csv(os.path.join(tuned_model_path + "/en_dev_out_zero.tsv"), sep='\t', index=False)
    FINAL_DF.to_csv(os.path.join(tuned_model_path + "/en_dev_out_zero.csv"), index=False)
    EVAL_INS = Evaluation(ARGS, FINAL_DF, os.path.join(tuned_model_path + "/en_dev_out_zero.json"))
    EVAL_INS.main()
    print("Out df is saved")
