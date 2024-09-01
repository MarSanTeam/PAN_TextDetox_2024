import os

import pandas as pd
from tqdm import tqdm

from detox.Evaluation import Evaluation
from detox.config import BaseConfig

tqdm.pandas()

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    # --------------------------------------------- Generate Prediction -----------------------------------------------
    base_model_path = "/mnt/disk2/LanguageModels/falcon-7b"
    tuned_model_path = "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/falcon"

    df_reference = pd.read_csv(
        "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Submission/sample_submission_dev.tsv", sep="\t")
    df_all = pd.read_csv("/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/zep2/en_dev_out.csv")
    # Load the data from the provided file paths
    df_reference_en = df_reference[df_reference["lang"] == "en"]

    FINAL_LIST = {"input": df_all["input"], "prediction": df_all["prediction"],
                  "reference": df_reference_en["neutral_sentence"].reset_index(drop=True)}

    FINAL_DF = pd.DataFrame(FINAL_LIST)
    FINAL_DF.to_csv(os.path.join(tuned_model_path + "/en_dev_out_zero.tsv"), sep='\t', index=False)
    FINAL_DF.to_csv(os.path.join(tuned_model_path + "/en_dev_out_zero.csv"), index=False)
    EVAL_INS = Evaluation(ARGS, FINAL_DF, os.path.join(tuned_model_path + "/en_dev_out_zero.json"))
    EVAL_INS.main()
    print("Given data is evaluated!")
