DEFAULT_INPUT_MODEL = "EleutherAI/pythia-6.9b"
SUGGESTED_INPUT_MODELS = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b"
]
DEFAULT_TRAINING_DATASET = "databricks/databricks-dolly-15k"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)
DETOX_INSTRUCTION = (

    "You’re an advanced AI model which can find a toxic piece of text, re-write only the toxic phrase in a non-toxic way while saving the main content as much as possible. "
    "Please do not add any extra information except the paraphrase of the toxic phrase and preserve the main content. "
    "try to generate no length more tha given toxic sample. "
    "The goal is to generate a detoxified version of the text without significantly altering the original content. Your task is to process toxic sentences in English and Russian. "
    # "For example, if the toxic sentence is 'He had steel b*lls too!' the detoxified version would be 'He was brave too!'."
    "Your answer should come after '### Detoxified: ' and add ### End. in the end of generation.")

DETOX_Content_KEY = "### InputContent: "
DETOX_Extract_KEY = "### Detoxified: "
DETOX_END_KEY = " ### End. "
DETOX_INS_KEY = "### Instruction: "
DETOX_RES_KEY = "### Response: "