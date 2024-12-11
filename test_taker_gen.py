"""This file contains the code through which the QA model takes the test and produces its confidence predictions, using a zero-shot generation pipeline.
The output should be a .csv with model predictions that can be used to check the similarity between the model's performance and the human performance"""

# Importing the necessary libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from comet import download_model, load_from_checkpoint
from scipy.stats import entropy
import torch
import itertools
from tqdm import tqdm
import argparse
import numpy as np
import os
import warnings
import logging


MODEL_NAME_DICT = {
    'Llama3-8b-chat': 'unsloth/llama-3-8b-Instruct-bnb-4bit',
    'Llama3-70b-chat': 'unsloth/llama-3-70b-Instruct-bnb-4bit',
    'Yi-34b-chat': 'unsloth/yi-34b-chat-bnb-4bit',
    'Mistral-7b-chat': 'unsloth/mistral-7b-instruct-v0.2-bnb-4bit',

    'Llama3-8b': 'unsloth/llama-3-8b-bnb-4bit',
    'Llama3-70b': 'unsloth/llama-3-70b-bnb-4bit',
    'Yi-34b': 'unsloth/yi-34b-bnb-4bit',
    'Mistral-7b': 'unsloth/mistral-7b-v0.2-bnb-4bit',

}

MODEL_NAME = ""
NUM_DISTRACTORS = 0
DATASET_NAME = ""
OUTPUT_FILE_NAME = ""
KEEP_ONLY = None
WITHOUT_QUESTION = False
WITHOUT_CHOICES = False
MODEL_INSTALLATION_FOLDER = ""
HF_TOKEN = ""
BOT_ROLE = ""
PROMPT_STYLE = ""
USE_COMET = False
WARNINGS = False


def parse_args():
    print("---------------PARSING ARGUMENTS---------------")
    parser = argparse.ArgumentParser(
        description='Perform text generation and Q&A tasks via Hugging Face models.')
    parser.add_argument('-m', '--model', type=str,
                        help='Model Name', required=True)
    parser.add_argument('-d', '--distractors', type=int,
                        help='Whether to use the 2 or 3 distractors dataset', required=True)
    parser.add_argument('-r', '--role', type=str,
                        help='Role of the chatbot', default="assistant")
    parser.add_argument('-p', '--prompt', type=int,
                        help='Which prompt formulation to use, 1 or 2', default=1)
    parser.add_argument('-o', '--output_name', type=str,
                        help='Output file name')
    parser.add_argument('-wq', '--without_question', action="store_true",
                        help='Whether to include the question in the prompt')
    parser.add_argument('-wc', '--without_choices', action="store_true",
                        help='Whether to include the question in the prompt')
    parser.add_argument('-comet', '--comet', action="store_true",
                        help='If enabled, uses comet for answer evaluation of open questions', default=False)
    parser.add_argument('-t', '--test_mode', action="store_true",
                        help='Test mode only uses five questions', default=False)
    parser.add_argument('-w', '--warnings', action="store_true",
                        help='If enabled, certain warnings are shown', default=False)

    return dict(vars(parser.parse_args()))


def set_globals(args):
    """Sets the global variables based on the arguments. There are settings for experiments not presented in the paper."""

    globals()["MODEL_NAME"] = MODEL_NAME_DICT[args['model']]
    globals()["NUM_DISTRACTORS"] = args['distractors']
    globals()["DATASET_NAME"] = "model_input/" + \
        str(NUM_DISTRACTORS) + "_distractors_processed.csv"
    globals()["WARNINGS"] = args['warnings']
    globals()["BOT_ROLE"] = args['role']
    globals()["PROMPT_STYLE"] = args['prompt']
    globals()["USE_COMET"] = args['comet']

    # Whether question and choices were provided, as well as whether the test is in test mode
    output_file_prefix = ""
    if args['test_mode']:
        globals()["KEEP_ONLY"] = 5
        output_file_prefix += "TEST_"

    if not args['without_question']:
        globals()["WITHOUT_QUESTION"] = False
        output_file_prefix += "Q"
    else:
        globals()["WITHOUT_QUESTION"] = True

    if not args['without_choices']:
        globals()["WITHOUT_CHOICES"] = False
        output_file_prefix += "C"
    else:
        globals()["WITHOUT_CHOICES"] = True

    if PROMPT_STYLE == 2:
        output_file_prefix += "PTWO"

    output_file_prefix += BOT_ROLE

    if args['output_name']:
        globals()["OUTPUT_FILE_NAME"] = "model_results/" + \
            args['output_name'] + ".csv"
    else:
        globals()["OUTPUT_FILE_NAME"] = "model_results/" + \
            str(NUM_DISTRACTORS) + "_distractors_" + output_file_prefix + \
            "_" + args['model'] + ".csv"

    print("Running with specs:")
    print("Test mode: ", args['test_mode'])
    print("Model: ", MODEL_NAME)
    print("Dataset: ", DATASET_NAME)
    print("Bot role: ", BOT_ROLE)
    print("Prompt style: ", PROMPT_STYLE)
    print("Without question: ", WITHOUT_QUESTION)
    print("Without choices: ", WITHOUT_CHOICES)
    print("Use Comet: ", USE_COMET)
    print("Show Warnings: ", WARNINGS)
    print("Output file: ", OUTPUT_FILE_NAME)
    if WITHOUT_QUESTION and WITHOUT_CHOICES:
        print("Error: Cannot run without question and choices")
        exit(1)

    globals()["HF_TOKEN"] = load_hf_token("../hf_token.txt")


def set_warnings():
    """Sets the warnings to be shown or not"""
    if not WARNINGS:
        warnings.filterwarnings("ignore", category=UserWarning)
        # Only show errors. Default is to show warnings and errors.
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        # For tensorflow: Hides INFO messages
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Disable lightning warnings, only relevant when comet is used.
        os.environ['POSSIBLE_USER_WARNINGS'] = 'off'
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.ERROR)
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
            logging.ERROR)


def load_hf_token(file_path):
    """Loads the Hugging Face token from a file"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return str(content).strip()
    except FileNotFoundError:
        print(
            f"Warning: HF token: The file '{file_path}' does not exist.")
        return None


class Generator():
    """This class is responsible for generating the model's responses and calculating the probabilities of the answer choices"""

    def __init__(self):
        print("----------INITIALISING GENERATOR MODEL---------")
        # The quantization config is ignored for models that are pre-quantized, e.g., Llama 3. A warning will pop up, but it can be ignored.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", quantization_config=quantization_config,  token=HF_TOKEN)

        if WITHOUT_CHOICES:
            print("----------INITIALISING SIMILARITY SCORER-------")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids a warning
            self.similarity_scorer = None
            if USE_COMET:
                comet_model = download_model("Unbabel/wmt22-cometkiwi-da")
                self.similarity_scorer = load_from_checkpoint(comet_model)
            else:
                self.similarity_scorer = SentenceTransformer(
                    "all-MiniLM-L6-v2")

        print("-------------INITIALISING TOKENIZER------------")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side="left", token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find index of each answer choice in the vocabulary. Done here to avoid doing it for each question. Remove the nans
        self.answer_choices_indices_a = self.tokenizer.convert_tokens_to_ids(
            ['A', ' A', 'A ', ' A ', 'a', ' a', 'a ', ' a '])
        self.answer_choices_indices_a = [
            x for x in self.answer_choices_indices_a if x is not None]
        self.answer_choices_indices_b = self.tokenizer.convert_tokens_to_ids(
            ['B', ' B', 'B ', ' B ', 'b', ' b', 'b ', ' b '])
        self.answer_choices_indices_b = [
            x for x in self.answer_choices_indices_b if x is not None]
        self.answer_choices_indices_c = self.tokenizer.convert_tokens_to_ids(
            ['C', ' C', 'C ', ' C ', 'c', ' c', 'c ', ' c '])
        self.answer_choices_indices_c = [
            x for x in self.answer_choices_indices_c if x is not None]
        if NUM_DISTRACTORS == 3:
            self.answer_choices_indices_d = self.tokenizer.convert_tokens_to_ids(
                ['D', ' D', 'D ', ' D ', 'd', ' d', 'd ', ' d '])  # Only used if NUM_DISTRACTORS == 3
            self.answer_choices_indices_d = [
                x for x in self.answer_choices_indices_d if x is not None]

    def open_qa_get_answer(self, question, choices):
        def get_similarity_scores(model_answer, choices):
            if USE_COMET:
                pairs_to_check = [{"src": model_answer, "mt": choice}
                                  for choice in choices]
                model_output = self.similarity_scorer.predict(
                    pairs_to_check, batch_size=1, progress_bar=False)
                scores = model_output['scores']

            else:
                model_answer_embeddings = self.similarity_scorer.encode(
                    model_answer, convert_to_tensor=True)

                choices_embeddings = self.similarity_scorer.encode(
                    choices, convert_to_tensor=True)

                # Compute cosine-similarities
                scores = util.cos_sim(
                    model_answer_embeddings, choices_embeddings)[0].tolist()

            # Normalize the scores
            normalised_scores = [score/sum(scores) for score in scores]
            return normalised_scores

        # 1st, prepare the prompt
        system_message = "You are a " + BOT_ROLE + "."
        chat = None
        # Gemma does not accept the system message
        if 'gemma' in MODEL_NAME or ('chat' not in MODEL_NAME):
            chat = [{"role": "user", "content": question}]
        else:
            chat = [{"role": "system", "content": system_message},
                    {"role": "user", "content": question}]
        processed_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")

        # 2nd, prepare the model inputs
        model_inputs = self.tokenizer(
            processed_prompt, return_tensors="pt", padding=True).to("cuda")

        # 3rd, generate the output
        # Since we have do_sample=False, the function ignores the temperature and top_p arguments
        output = self.model.generate(**model_inputs, max_new_tokens=60, do_sample=False,
                                     output_scores=True, num_return_sequences=1, return_dict_in_generate=True,
                                     renormalize_logits=False, temperature=None, top_p=None, pad_token_id=self.tokenizer.eos_token_id)

        output_text = self.tokenizer.decode(
            output.sequences[0], skip_special_tokens=True)

        output_text = output_text.split("Response:")[1].strip()
        output_text = output_text.replace(BOT_ROLE, "")
        output_text = output_text.replace("assistant", "")
        output_text = output_text.replace("\n", "")
        scores = get_similarity_scores(output_text, choices)
        return scores, output_text

    def calc_probabilities_for_prompt(self, question):
        prompt = None
        if 'chat' in MODEL_NAME:
            # 1st, prepare the prompt
            system_message = "You are a " + BOT_ROLE + "."
            chat = None
            if 'gemma' in MODEL_NAME:  # Gemma models cannot use the system message
                chat = [{"role": "user", "content": question}]
            else:
                chat = [{"role": "system", "content": system_message},
                        {"role": "user", "content": question}]

            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        else:
            prompt = question

        # 2nd, prepare the model inputs
        model_inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True).to("cuda")

        # 3rd, generate the output
        # Since we have do_sample=False, the function ignores the temperature and top_p arguments
        output = self.model.generate(**model_inputs, max_new_tokens=30, do_sample=False,
                                     output_scores=True, num_return_sequences=1, return_dict_in_generate=True,
                                     renormalize_logits=False, temperature=None, top_p=None, pad_token_id=self.tokenizer.eos_token_id)

        # 4th, get the probabilities of the next token
        next_token_candidates_tensor = output.scores[0][0]
        # next_token_candidates_tensor should hold the probability of each vocabulary token being the next token
        # Scale the probabilities to sum to 1
        probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=0)

        # Convert probabilities to list and scale to 1. Get the max probability of the relevant tokens
        probs = [0] * (NUM_DISTRACTORS + 1)  # +1 for the correct answer
        probs[0] = max([probabilities[idx].item()
                       for idx in self.answer_choices_indices_a])
        probs[1] = max([probabilities[idx].item()
                       for idx in self.answer_choices_indices_b])
        probs[2] = max([probabilities[idx].item()
                       for idx in self.answer_choices_indices_c])
        if NUM_DISTRACTORS == 3:
            probs[3] = max([probabilities[idx].item()
                           for idx in self.answer_choices_indices_d])

        total = sum(probs)
        probs = [p / total for p in probs]

        output_text = self.tokenizer.decode(
            output.sequences[0], skip_special_tokens=True)
        # Only keep the part after "Response:"
        output_text = output_text.split(
            "Response:")[1].strip()
        output_text = output_text.replace(BOT_ROLE, "")
        output_text = output_text.replace("assistant", "")
        output_text = output_text.replace("\n", "")

        return probs, output_text


def format_question_string(question, choices):
    # Given a question and a list of answer choices, returns a string with the question and the answer choices
    prompt = ""
    if not WITHOUT_QUESTION and not WITHOUT_CHOICES:
        question_string = question + "\n A:" + choices[0] + "\n B:" + \
            choices[1] + "\n C:" + choices[2]
        if NUM_DISTRACTORS == 3:
            question_string += ("\n D:" + choices[3])
        if PROMPT_STYLE == 1:
            prompt = f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer with no explanation.

            Question:

            {question_string}

            Response:\n
            """
        elif PROMPT_STYLE == 2:
            prompt = f"""You will be presented with a multiple-choice question. Select the option letter that you believe provides the best answer to the question. Keep your response concise by simply stating the letter of your chosen answer without providing any additional explanation.

            Question:

            {question_string}

            Response:\n
            """
    elif WITHOUT_QUESTION and not WITHOUT_CHOICES:
        question_string = "\n A:" + choices[0] + "\n B:" + \
            choices[1] + "\n C:" + choices[2]
        if NUM_DISTRACTORS == 3:
            question_string += ("\n D:" + choices[3])
        if PROMPT_STYLE == 1:
            prompt = f"""Below are options to a multiple-choice question, without the question. Choose the letter that corresponds to the correct answer. Keep your response as brief as possible; just state the letter corresponding to your answer with no explanation.

                Choices:

                {question_string}

                Response:\n
                """
        elif PROMPT_STYLE == 2:
            prompt = f"""You will be presented a list of options to a multiple-choice question, without the question. Select the option letter that you believe corresponds to the corrent answer. Keep your response concise by simply stating the letter of your chosen answer without providing any additional explanation.

                Choices:

                {question_string}

                Response:\n
                """
    elif not WITHOUT_QUESTION and WITHOUT_CHOICES:
        if PROMPT_STYLE == 1:
            prompt = f"""Answer the following question. Keep your response as brief as possible, with no explanation.

            Question:

            {question}

            Response:\n
            """
        elif PROMPT_STYLE == 2:
            prompt = f""" You will be presented an open question. Answer the question as concisely as possible, without providing any additional explanation.

            Question:

            {question}

            Response:\n
            """
    return prompt


def calc_avg_prob_per_answer_choice(row, generator):
    def get_prompt_permutations(question, choices):
        # Returns a list of all possible questions with the different permutations of the answer choices
        # List of permutations and for each, the original order of the answer choices
        prompts = []
        original_order = []
        for perm in itertools.permutations(choices):
            prompts.append(format_question_string(question, perm))
            original_order.append([choices.index(p) for p in perm])
        return prompts, original_order
    # Given a row of the questions_set dataframe, returns the average probability for each answer choice
    question = row['question']
    choices = [row['correct_answer'], row['distractor1'],
               row['distractor2']]
    probs = []
    eventual_choices = []
    std_eventual_choices_order = []
    probability_eventual_choices_order = []
    full_answer = ""  # in case of permutations, only the 1st one is kept
    full_prompt = ""  # in case of permutations, only the 1st one is kept
    if NUM_DISTRACTORS == 3:
        choices.append(row['distractor3'])

    if WITHOUT_CHOICES:  # open QA case
        full_prompt = format_question_string(question, choices)
        probs, full_answer = generator.open_qa_get_answer(
            full_prompt, choices)
        # No ordering effect here.
        std_eventual_choices_order = [0.0]*(NUM_DISTRACTORS+1)
        probability_eventual_choices_order = [0.0]*(NUM_DISTRACTORS+1)

    else:
        prompts, original_order = get_prompt_permutations(question, choices)
        for i, p in enumerate(tqdm(prompts, desc="Calculating probabilities", leave=False)):
            order_for_this_prompt = original_order[i]
            probs_not_in_original_order, full_answer_temp = generator.calc_probabilities_for_prompt(
                p)
            if full_answer == "":
                full_answer = full_answer_temp
            # original_order is a list of the original order of the answer choices, e.g., [1,3,0,2]
            # probs_not_in_original_order is a list of probabilities, e.g., [0.5, 0.32, 0.53, 0.24]
            # reorder the probabilities to match the original order of the answer choices. 0.53, 0.5, 0.24, 0.32
            reordered_probs = [0] * len(probs_not_in_original_order)
            reordered_eventual_choices = [0] * len(reordered_probs)
            for j, idx in enumerate(order_for_this_prompt):
                reordered_probs[idx] = probs_not_in_original_order[j]
            # based on reordered_probs, the eventual_choices is just the maximum element turned to one. For example, from [0.4, 0.3, 0.2, 0.1] to [1, 0, 0, 0]
            reordered_eventual_choices[reordered_probs.index(
                max(reordered_probs))] = 1
            # Store the results of this ordering
            eventual_choices.append(reordered_eventual_choices)
            probs.append(reordered_probs)

        # We calculate the standard deviation for each answer choice
        std_eventual_choices_order = [
            np.std(x) for x in zip(*eventual_choices)]
        probability_eventual_choices_order = [np.mean(x) for x in zip(
            *eventual_choices)]

        # We calculate the average probability for each answer choice
        probs = [sum(x) / len(probs) for x in zip(*probs)]
        full_prompt = prompts[0]  # Only the first prompt is kept
    # option probabilities, ordering st. deviation, ordering probabilities, full generated answer and full prompt
    return probs, std_eventual_choices_order, probability_eventual_choices_order, full_answer, full_prompt


# Main function
if __name__ == "__main__":
    set_globals(parse_args())
    set_warnings()
    # Make sure there is a model_results folder
    if not os.path.exists("model_results"):
        os.makedirs("model_results")

    # Read data
    questions_set = pd.read_csv(DATASET_NAME)

    # Remove questions where is_extra is true. These questions contain corrupted/incomplete/with images data
    questions_set = questions_set[questions_set['is_extra'] == False]

    if KEEP_ONLY:
        questions_set = questions_set.head(KEEP_ONLY)
    generator = Generator()

    print("--------------EXECUTING INFERENCE--------------")
    # Calculate model uncertainty using the different metrics
    for index, row in tqdm(list(questions_set.iterrows()), total=len(questions_set)):
        probs, order_eventual_choices_stdev, order_eventual_choices_probability, full_answer, full_prompt = calc_avg_prob_per_answer_choice(
            row, generator)

        # STORE FULL PROMPT AND ANSWER
        questions_set.at[index, 'full_answer'] = full_answer
        questions_set.at[index, 'full_prompt'] = full_prompt

        # STORE ORDER SENSITIVITY METRICS: STDEV
        questions_set.at[index,
                         'correct_answer_order_std'] = order_eventual_choices_stdev[0]
        questions_set.at[index,
                         'distractor1_order_std'] = order_eventual_choices_stdev[1]
        questions_set.at[index,
                         'distractor2_order_std'] = order_eventual_choices_stdev[2]
        if NUM_DISTRACTORS == 3:
            questions_set.at[index,
                             'distractor3_order_std'] = order_eventual_choices_stdev[3]

        # STORE ORDER SENSITIVITY METRICS: PROBABILITY
        questions_set.at[index,
                         'correct_answer_order_probability'] = order_eventual_choices_probability[0]
        questions_set.at[index,
                         'distractor1_order_probability'] = order_eventual_choices_probability[1]
        questions_set.at[index,
                         'distractor2_order_probability'] = order_eventual_choices_probability[2]
        if NUM_DISTRACTORS == 3:
            questions_set.at[index,
                             'distractor3_order_probability'] = order_eventual_choices_probability[3]

        # STORE LOGIT PROBS METRICS
        questions_set.at[index, 'correct_answer_model_rate'] = probs[0]
        questions_set.at[index, 'distractor1_model_rate'] = probs[1]
        questions_set.at[index, 'distractor2_model_rate'] = probs[2]
        if NUM_DISTRACTORS == 3:
            questions_set.at[index, 'distractor3_model_rate'] = probs[3]

        # CALCULATE model_choices_inverse_entropy (this is not about order):
        questions_set.at[index, 'model_choices_entropy'] = entropy(probs)
        questions_set.at[index,
                         'model_choices_inverse_entropy'] = 1 - entropy(probs)  # can be negative

    # create model_is_correct column
    if NUM_DISTRACTORS == 2:
        questions_set['model_is_correct'] = questions_set['correct_answer_model_rate'] == \
            questions_set[['correct_answer_model_rate',
                           'distractor1_model_rate', 'distractor2_model_rate']].max(axis=1)
    else:
        questions_set['model_is_correct'] = questions_set['correct_answer_model_rate'] == \
            questions_set[['correct_answer_model_rate', 'distractor1_model_rate',
                           'distractor2_model_rate', 'distractor3_model_rate']].max(axis=1)

    print("---------------EXPORTING RESULTS---------------")
    # Export the results
    questions_set.to_csv(OUTPUT_FILE_NAME, index=False)
    print("Output file saved as ", OUTPUT_FILE_NAME)
    print("---------------------DONE!---------------------")
