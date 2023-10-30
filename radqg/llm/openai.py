##########################################################################################
# Description: A script containing general functionalites for working with OpenAI API.
##########################################################################################

import openai
import radqg.configs as configs
from radqg.prompts import get_qa_prompt, edit_qa_prompt, get_dict_formatting_prompt

# ----------------------------------------------------------------------------------------
# Configurations

openai.api_key = configs.OPENAI_API_KEY

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
) -> dict:
    """A function to generate a question from a figure caption using the OpenAI LLMs."""

    # Asking for the initial question and answer generation
    prompt1 = get_qa_prompt(
        figure_caption=caption,
        context=context,
        type_of_question=type_of_question,
    )
    good_question_created = False
    while True:
        message1 = [{"role": "user", "content": prompt1}]
        response1 = openai.ChatCompletion.create(
            model=configs.OPENAI_QA_GEN_MODEL,
            messages=message1,
            temperature=0.6,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        out_dict_string1 = response1.choices[0]["message"]["content"]

        # Asking for double-checking the question and answer generation
        prompt2 = edit_qa_prompt(caption, out_dict_string1)
        message2 = [{"role": "user", "content": prompt2}]
        response2 = openai.ChatCompletion.create(
            model=configs.OPENAI_QA_EDIT_MODEL,
            messages=message2,
            temperature=0.6,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        out_dict_string2 = response2.choices[0]["message"]["content"]

        # Asking for the dictionary formatting
        out_dict_string3 = out_dict_string2
        count_revised = 0
        while count_revised < 5:
            prompt3 = get_dict_formatting_prompt(out_dict_string3)
            message3 = [{"role": "user", "content": prompt3}]
            response3 = openai.ChatCompletion.create(
                model=configs.OPENAI_FORMATTING_MODEL,
                messages=message3,
                temperature=0.2,
                max_tokens=2000,
                frequency_penalty=0.0,
            )
            out_dict_string3 = response3.choices[0]["message"]["content"]
            count_revised += 1
            try:
                qa_dict = eval(out_dict_string3)
                assert isinstance(qa_dict, dict)
                good_question_created = True
                break
            except:
                print("The following string is not a valid Python dictionary:")
                continue
        if good_question_created:
            return qa_dict
