##########################################################################################
# Description: A script containing general functionalites for working with OpenAI API.
##########################################################################################

import openai
import radqg.configs as configs
from radqg.prompts import get_generator_prompt
from radqg.prompts import get_contenteditor_prompt, get_formateditor_prompt
from radqg.utils import count_tokens

# ----------------------------------------------------------------------------------------
# Configurations

openai.api_key = configs.OPENAI_API_KEY
price_for_gpt4_tokens: float = 0.03 / 1000  # $0.03 per 1000 tokens
price_for_chatgpt_tokens: float = 0.0015 / 1000  # $0.01 per 1000 tokens

# ----------------------------------------------------------------------------------------
# get_price_for_tokens

def get_price_for_tokens(total_tokens: int, model: str) -> float:
    """A function to calculate the price for a given number of tokens."""
    
    if model == "gpt-4":
        return total_tokens * price_for_gpt4_tokens
    elif model == "gpt-3.5-turbo":
        return total_tokens * price_for_chatgpt_tokens
    else:
        raise ValueError("The model is not supported.")

# ----------------------------------------------------------------------------------------
# emb_fn


def embed_fn(
    text_list_to_embed: list[str], model: str = configs.OPENAI_EMBEDDING_MODEL
) -> list[float]:
    """A function to embed a list of texts using OpenAI API."""

    response = openai.Embedding.create(model=model, input=text_list_to_embed)
    embeddings = [
        response["data"][i]["embedding"] for i in range(len(response["data"]))
    ]

    return embeddings


# ----------------------------------------------------------------------------------------
# qa


def qa(
    caption: str,
    context: str,
    type_of_question: str,
    generator_model: str = configs.OPENAI_GENERATOR_MODEL,
    content_editor_model: str = configs.OPENAI_CONTENT_EDITOR_MODEL,
    format_editor_model: str = configs.OPENAI_FORMAT_EDITOR_MODEL,
) -> dict:
    """A function to generate a question from a figure caption using the OpenAI LLMs."""

    # To check the total number of tokens and budget used.
    total_tokens = 0
    total_cost = 0
    
    # Asking for the initial question and answer generation
    prompt1 = get_generator_prompt(
        figure_caption=caption,
        context=context,
        type_of_question=type_of_question,
    )
    good_question_created = False
    while True:
        message1 = [{"role": "user", "content": prompt1}]
        response1 = openai.ChatCompletion.create(
            model=generator_model,
            messages=message1,
            temperature=0.6,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        out_dict_string1 = response1.choices[0]["message"]["content"]
        total_tokens += count_tokens(prompt1) + count_tokens(out_dict_string1)
        total_cost += get_price_for_tokens(total_tokens, model=configs.OPENAI_GENERATOR_MODEL)

        # Asking for double-checking the question and answer generation
        prompt2 = get_contenteditor_prompt(caption, out_dict_string1)
        message2 = [{"role": "user", "content": prompt2}]
        response2 = openai.ChatCompletion.create(
            model=content_editor_model,
            messages=message2,
            temperature=0.6,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        out_dict_string2 = response2.choices[0]["message"]["content"]
        total_tokens += count_tokens(prompt2) + count_tokens(out_dict_string2)
        total_cost += get_price_for_tokens(total_tokens, model=configs.OPENAI_CONTENT_EDITOR_MODEL)

        # Asking for the dictionary formatting
        out_dict_string3 = out_dict_string2
        count_revised = 0
        while count_revised < 5:
            prompt3 = get_formateditor_prompt(out_dict_string3)
            message3 = [{"role": "user", "content": prompt3}]
            response3 = openai.ChatCompletion.create(
                model=format_editor_model,
                messages=message3,
                temperature=0.2,
                max_tokens=2000,
                frequency_penalty=0.0,
            )
            out_dict_string3 = response3.choices[0]["message"]["content"]
            count_revised += 1
            total_tokens += count_tokens(prompt3) + count_tokens(out_dict_string3)
            total_cost += get_price_for_tokens(total_tokens, model=configs.OPENAI_FORMAT_EDITOR_MODEL)
            try:
                qa_dict = eval(out_dict_string3)
                assert isinstance(qa_dict, dict)
                good_question_created = True
                break
            except:
                print("The following string is not a valid Python dictionary:")
                print(out_dict_string3)
                continue
        if good_question_created:
            return qa_dict, out_dict_string1, out_dict_string2, out_dict_string3, total_tokens, total_cost
