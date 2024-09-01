import matplotlib.pyplot as plt
import pandas as pd

from detox import LanguageModelLoader
from detox.config.config import ModelArguments, BaseConfig

# Load your Excel file
df = pd.read_excel('SPV2-train.xlsx')
if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    # Initialize BERT tokenizer
    MODE = "test"
    LM_LOADER = LanguageModelLoader(ModelArguments.model_name_or_path, MODE, ARGS)
    # --------------------------------------------- Run model -----------------------------------------------
    LM_LOADER.create_bnb_config()
    BASE_MODEL, TOKENIZER = LM_LOADER.setup_model()
    # Tokenize the 'answer' column and get token lengths
    # tokenized_data = df['0.answer'].apply(lambda x: len(TOKENIZER.encode(x, add_special_tokens=True)))
    tokenized_data = df['0.answer'].apply(lambda x: len(TOKENIZER.tokenize(x, add_special_tokens=True)))
    # df['answer'].apply(lambda x: len(TOKENIZER.tokenize(x)))
    # Plot the distribution of token lengths
    plt.figure(figsize=(10, 6))
    plt.hist(tokenized_data, bins=40, color='red', edgecolor='white')
    plt.title('Distribution of Tokenized Data Lengths of SPV2')
    plt.xlabel('Tokenized Data Length')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig("SPV2.png")
